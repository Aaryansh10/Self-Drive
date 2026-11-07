#include <cstdlib>
#include <iostream>
#include "bt/nodes.hpp"

#define CHECK(expr) do { if (!(expr)) { std::cerr << "FAIL " << __LINE__ << ": " #expr "\n"; std::exit(1); } } while (0)
using Status = BT::NodeStatus;

bt::DecisionConfig config() {
    bt::DecisionConfig cfg;
    cfg.min_distance = 2.0;
    cfg.confirm_stop = 1;
    cfg.classes["pedestrian"] = {bt::Action::STOP, 5.0};
    cfg.classes["stopline"] = {bt::Action::INTERSECTION, 3.96};
    cfg.classes["sign"] = {bt::Action::NONE, 8.0, true};
    return cfg;
}

bt::Detection detection(const char* cls, double distance = 3.0) {
    bt::Detection d;
    d.cls = cls; d.valid = true; d.score = 1; d.n_pts = 10;
    d.distance = distance; d.x = 10;
    return d;
}

struct Scenario {
    double time = 1.0;
    std::shared_ptr<bt::SharedState> state = std::make_shared<bt::SharedState>(config(), 1.0, [this] { return time; });
    int stops = 0, turns = 0, cancels = 0;
    bool lane = false;
    Status stop_result = Status::RUNNING, turn_result = Status::RUNNING;
    Scenario() { state->set_turns({1, 0}); }
    void line(bool real_sign, double distance = 3.0) {
        std::vector<bt::Detection> dets{detection("stopline", distance)};
        if (real_sign) dets.push_back(detection("sign", distance));
        time += 0.1;
        state->on_intersection_detections(dets);
    }
};

class StopStub : public BT::StatefulActionNode {
public:
    StopStub(const std::string& n, const BT::NodeConfig& c, Scenario* s) : BT::StatefulActionNode(n,c), s_(s) {}
    static BT::PortsList providedPorts() { return {BT::InputPort<double>("distance")}; }
    Status onStart() override {
        ++s_->stops;
        CHECK(getInput<double>("distance").value() == s_->state->current().distance);
        return s_->stop_result;
    }
    Status onRunning() override { return s_->stop_result; }
    void onHalted() override {}
private:
    Scenario* s_;
};
class TurnStub : public BT::StatefulActionNode {
public:
    TurnStub(const std::string& n, const BT::NodeConfig& c, Scenario* s) : BT::StatefulActionNode(n,c), s_(s) {}
    static BT::PortsList providedPorts() { return {}; }
    Status onStart() override {
        ++s_->turns;
        CHECK(!s_->state->active_intersection().needs_stop);
        CHECK(!s_->lane);
        return result();
    }
    Status onRunning() override { return result(); }
    void onHalted() override { ++s_->cancels; }
private:
    Status result() {
        if (s_->turn_result == Status::SUCCESS)
            s_->state->complete_intersection(s_->state->active_intersection());
        return s_->turn_result;
    }
    Scenario* s_;
};
class LaneStub : public BT::StatefulActionNode {
public:
    LaneStub(const std::string& n, const BT::NodeConfig& c, Scenario* s) : BT::StatefulActionNode(n,c), s_(s) {}
    static BT::PortsList providedPorts() { return {}; }
    Status onStart() override { s_->lane=true; return Status::RUNNING; }
    Status onRunning() override { return Status::RUNNING; }
    void onHalted() override { s_->lane=false; }
private:
    Scenario* s_;
};

BT::Tree tree_for(Scenario& s) {
    BT::BehaviorTreeFactory f;
    f.registerNodeType<bt::DecideAction>("DecideAction", s.state);
    f.registerNodeType<bt::BeginIntersection>("BeginIntersection", s.state);
    f.registerNodeType<bt::MarkIntersectionStopped>("MarkIntersectionStopped", s.state);
    f.registerNodeType<bt::RetryOnNextTick>("RetryOnNextTick");
    f.registerNodeType<StopStub>("StopRobot", &s);
    f.registerNodeType<TurnStub>("Intersection", &s);
    f.registerNodeType<LaneStub>("LaneFollowing", &s);
    f.registerSimpleAction("DisableLaneFollowing", [&s](BT::TreeNode&) { s.lane=false; return Status::SUCCESS; });
    f.registerSimpleAction("LaneChange", [](BT::TreeNode&) { return Status::SUCCESS; }, {BT::InputPort<int>("obstacle_id")});
    return f.createTreeFromFile(TEST_TREE);
}

int main() {
    { // Explicit real sign stops first, survives stale input, and turns in the success tick.
        Scenario s;
        auto tree = tree_for(s);
        tree.tickOnce(); CHECK(s.lane);
        for (int i=0;i<3;++i) s.line(true);
        tree.tickOnce(); CHECK(s.stops==1 && s.turns==0 && !s.lane);
        s.time += 25;
        tree.tickOnce(); CHECK(s.stops==1 && s.turns==0);
        s.stop_result=Status::SUCCESS;
        tree.tickOnce(); CHECK(s.turns==1 && s.stops==1);
        s.turn_result=Status::FAILURE;
        tree.tickOnce(); CHECK(!s.lane);
        s.turn_result=Status::RUNNING;
        tree.tickOnce(); CHECK(s.turns==2 && s.stops==1);
        s.turn_result=Status::SUCCESS;
        tree.tickOnce();
        uint8_t turn=9; CHECK(s.state->peek_turn(turn) && turn==0);
        tree.tickOnce(); CHECK(s.lane);
        s.time += 5; // Longer than the generic id-less obstacle TTL.
        for (int i=0;i<3;++i) s.line(false);
        tree.tickOnce(); CHECK(s.turns==2); // Same physical line is handled.
        tree.haltTree();
    }
    { // No real-sign marker (including fake/unknown sign): direct intersection, even near the line.
        Scenario s;
        auto tree=tree_for(s);
        for (int i=0;i<3;++i) s.line(false, 0.25);
        tree.tickOnce(); CHECK(s.stops==0 && s.turns==1);
        // A pedestrian still cancels the maneuver; resumption retains its turn.
        s.state->on_detections({detection("pedestrian")});
        tree.tickOnce(); CHECK(s.cancels==1 && s.stops==1);
        s.time += 2;
        tree.tickOnce(); CHECK(s.turns==2 && s.stops==1);
        tree.haltTree();
    }
    { // Fast obstacle messages must not confirm a single stop-line frame.
        Scenario s;
        s.line(false);
        for (int i=0;i<10;++i) s.state->on_detections({});
        CHECK(s.state->current().action==bt::Action::NONE);
        s.line(false); s.line(false);
        CHECK(s.state->current().action==bt::Action::INTERSECTION);
    }
    { // Original sign timestamp controls TTL, even when line callbacks repeat.
        bt::Decider d(config());
        auto sign=detection("sign"); sign.observed_at=1.0;
        for (int i=0;i<3;++i) d.update({sign,detection("stopline")}, 22.0+i*0.1);
        CHECK(!d.update({sign,detection("stopline")},23.0).needs_stop);
    }
    { // A current fake-sign result replaces earlier real-sign memory before entry.
        Scenario s;
        s.state->on_intersection_detections({detection("sign", 6.0)}, true);
        for (int i=0;i<3;++i) {
            s.time += 0.1;
            s.state->on_intersection_detections({detection("stopline", 2.5)}, true);
        }
        const auto d = s.state->current();
        CHECK(d.action == bt::Action::INTERSECTION && !d.needs_stop);
        CHECK(d.distance == 2.5);
    }
    std::cout << "Intersection decision and XML tests passed\n";
}
