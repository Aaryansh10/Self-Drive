// Plain-assert tests for decision.hpp (no ROS, no gtest). Run with ctest / colcon test, or build and execute directly.
#include <cstdio>
#include <cstdlib>

#include "bt/decision.hpp"

using namespace bt;

static int failures = 0;
static Decision last_dec;  // last decision returned by feed()
#define CHECK(cond)                                       \
  do {                                                    \
    if (!(cond)) {                                        \
      std::printf("FAIL line %d: %s\n", __LINE__, #cond); \
      ++failures;                                         \
    }                                                     \
  } while (0)

static DecisionConfig make_cfg() {
  DecisionConfig c;
  c.classes["pedestrians"] = {Action::STOP, 4.0};
  c.classes["stop_sign"] = {Action::STOP, 3.0};
  c.classes["barrels"] = {Action::CHANGE, 5.0};
  c.classes["tyre"] = {Action::CHANGE, 5.0};
  c.classes["potholes"] = {Action::CHANGE, 5.0};
  return c;
}

static Detection det(const char* cls, double dist, long id = -1, double score = 0.9) {
  Detection d;
  d.cls = cls; d.distance = dist; d.id = id; d.score = score; d.n_pts = 10; d.valid = true;
  return d;
}

// feed the same list n times, 0.1 s apart, return the last decision
static Decision feed(Decider& dc, const std::vector<Detection>& l, double& t, int n) {
  Decision d;
  for (int i = 0; i < n; ++i) { t += 0.1; d = dc.update(l, t); }
  last_dec = d;
  return d;
}

int main() {
  { // 1. barrel alone: confirmed after 3 lists
    Decider dc(make_cfg()); double t = 0;
    CHECK(feed(dc, {det("barrels", 4.2)}, t, 2).action == Action::NONE);
    CHECK(feed(dc, {det("barrels", 4.2)}, t, 1).action == Action::CHANGE);
  }
  { // 2. barrel + pedestrian both triggered: STOP wins
    Decider dc(make_cfg()); double t = 0;
    Decision d = feed(dc, {det("barrels", 3.4), det("pedestrians", 3.5)}, t, 3);
    CHECK(d.action == Action::STOP);
    CHECK(d.cls == "pedestrians");
  }
  { // 3. pedestrian outside its own trigger distance does not stop; barrel still changes
    Decider dc(make_cfg()); double t = 0;
    CHECK(feed(dc, {det("barrels", 3.4), det("pedestrians", 6.0)}, t, 3).action == Action::CHANGE);
  }
  { // 4. pothole at 7 m: nothing
    Decider dc(make_cfg()); double t = 0;
    CHECK(feed(dc, {det("potholes", 7.0)}, t, 5).action == Action::NONE);
  }
  { // 5. stop needs only 2 lists; invalid / unknown class / low score / too close are ignored
    Decider dc(make_cfg()); double t = 0;
    CHECK(feed(dc, {det("stop_sign", 2.0)}, t, 1).action == Action::NONE);
    CHECK(feed(dc, {det("stop_sign", 2.0)}, t, 1).action == Action::STOP);
    Decider d2(make_cfg()); t = 0;
    Detection bad = det("barrels", 3.0); bad.valid = false;
    CHECK(feed(d2, {bad, det("cone", 2.0), det("barrels", 3.0, -1, 0.1), det("barrels", 0.2)}, t, 5).action == Action::NONE);
  }
  { // 6. stop is held briefly, released after hold_time, then the barrel gives CHANGE
    Decider dc(make_cfg()); double t = 0;
    CHECK(feed(dc, {det("pedestrians", 3.0), det("barrels", 3.4)}, t, 3).action == Action::STOP);
    CHECK(feed(dc, {det("barrels", 3.4)}, t, 1).action == Action::STOP);
    CHECK(feed(dc, {det("barrels", 3.4)}, t, 6).action == Action::CHANGE);
  }
  { // 7. pseudo id: same barrel getting closer is NOT a new obstacle after handling
    Decider dc(make_cfg()); double t = 0;
    CHECK(feed(dc, {det("barrels", 3.0)}, t, 3).action == Action::CHANGE);
    dc.mark_handled(last_dec, t);
    CHECK(feed(dc, {det("barrels", 2.6)}, t, 1).action == Action::NONE);
    CHECK(feed(dc, {det("barrels", 2.0)}, t, 5).action == Action::NONE);
  }
  { // 8. pseudo id: a FARTHER barrel appears right after -> new change, no waiting
    Decider dc(make_cfg()); double t = 0;
    feed(dc, {det("barrels", 3.0)}, t, 3);
    dc.mark_handled(last_dec, t);
    CHECK(feed(dc, {det("barrels", 2.5), det("barrels", 4.8)}, t, 3).action == Action::CHANGE);
  }
  { // 9. real ids: same id skipped, new id triggers
    Decider dc(make_cfg()); double t = 0;
    CHECK(feed(dc, {det("barrels", 4.0, 7)}, t, 3).action == Action::CHANGE);
    dc.mark_handled(last_dec, t);
    CHECK(feed(dc, {det("barrels", 3.0, 7)}, t, 3).action == Action::NONE);
    Decision d = feed(dc, {det("barrels", 3.0, 7), det("barrels", 4.5, 8)}, t, 3);
    CHECK(d.action == Action::CHANGE);
    CHECK(d.obstacle_id == 8);
  }
  { // 10. handled memory expires: same-looking obstacle much later counts as new
    Decider dc(make_cfg()); double t = 0;
    feed(dc, {det("barrels", 3.0)}, t, 3);
    dc.mark_handled(last_dec, t);
    t += 5.0;
    CHECK(feed(dc, {det("barrels", 3.0)}, t, 3).action == Action::CHANGE);
  }
  { // 11. one missing list does not drop a confirmed action (hold_time)
    Decider dc(make_cfg()); double t = 0;
    feed(dc, {det("barrels", 3.0)}, t, 3);
    CHECK(feed(dc, {}, t, 1).action == Action::CHANGE);
  }
  { // 12. mark_handled marks exactly the obstacle the goal was sent for, not whatever is current now
    Decider dc(make_cfg()); double t = 0;
    Decision sent = feed(dc, {det("barrels", 4.0, 7)}, t, 3);   // goal sent for id 7
    CHECK(sent.obstacle_id == 7);
    feed(dc, {det("barrels", 4.0, 7), det("barrels", 2.0, 8)}, t, 1);  // id 8 is nearer by the time the result arrives
    dc.mark_handled(sent, t);
    Decision d = feed(dc, {det("barrels", 4.0, 7), det("barrels", 2.0, 8)}, t, 3);
    CHECK(d.action == Action::CHANGE);
    CHECK(d.obstacle_id == 8);  // 7 is handled, 8 still needs its own lane change
    dc.mark_handled(Decision{}, t);  // a non-CHANGE decision is ignored
  }
  std::printf(failures ? "%d FAILURE(S)\n" : "ALL PASSED\n", failures);
  return failures ? 1 : 0;
}