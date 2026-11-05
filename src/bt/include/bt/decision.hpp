// decides what the robot should do from one list of detections.
/*
 every incoming detection list:
1. drop invalid / low-score / unknown-class detections
2. keep only detections which are already inside their OWN class trigger distance
3. priority by action: STOP > CHANGE > NONE ( intersecn and parking )
4. an action must be seen in confirm consecutive lists before it is reported (stop reacts faster)
5. a CHANGE obstacle that was already handled by a lane change is skipped:
		- with a real obstacle id (id >= 0): same id = same obstacle
		- without one (id < 0): same class and a distance that did NOT jump up = same obstacle

		*/

#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace bt
{

	enum class Action
	{
		NONE = 0,
		CHANGE = 1,
		STOP = 2,
		INTERSECTION = 3
	};

	inline const char *to_string(Action a)
	{
		switch (a)
		{
		case Action::CHANGE:
			return "change";
		case Action::STOP:
			return "stop";
		case Action::INTERSECTION:
			return "intersection";
		default:
			return "none";
		}
	}

	inline int slot(Action a) {
		return a == Action::STOP ? 1 : (a == Action::INTERSECTION ? 2 : 0);
	}

	struct Detection
	{
		std::string cls;
		double score = 0.0;
		int n_pts = 0;
		bool valid = false;
		double distance = 0.0;
		long id = -1;
		double x = 0.0;		  // Global Odom X
		double y = 0.0;		  // Global Odom Y
		double local_y = 0.0; // Local Base Y for lateral filter
	};

	struct ClassRule
	{
		Action action = Action::NONE;
		double trigger_distance = 0.0; // m
		bool mark_stop_sign = false; // seeing this class sets the stop sign flag to true
	};

	struct DecisionConfig
	{
		std::map<std::string, ClassRule> classes;
		double min_score = 0.5;
		int min_points = 1;
		double min_distance = 0.5; // m, closer readings are treated as garbage
		double max_lateral_offset = 1.2;
		int confirm_change = 3;			// consecutive detection lists before CHANGE is reported
		int confirm_stop = 3;			// consecutive detection lists before STOP is reported
		int confirm_intersection = 3;	// consecutive detection lists before INTERSECTION is reported
		double hold_time = 0.5;			// s, keep reporting an action after its detection flickers out
		double handled_ttl = 30.0;		// s, forget a handled obstacle id after this long unseen
		double pseudo_ttl = 2.0;		// s, forget a handled id-less obstacle after this long unseen
		double same_obstacle_tol = 0.5; // m, distance may grow by this much and still be "the same obstacle"

		double intersection_hold = 20.0;	// INTERSECTION remains active for these many seconds or until handeled
		double stop_sign_ttl = 20.0;	// A seen stop sign lasts for these many seconds
	};

	struct Decision // Action to be performed
	{
		Action action = Action::NONE;
		long obstacle_id = -1;
		double distance = std::numeric_limits<double>::infinity();
		std::string cls;
		double x = 0.0;
		double y = 0.0;
		bool needs_stop = false;
	};

	class Decider
	{
	public:
		explicit Decider(DecisionConfig cfg) : cfg_(std::move(cfg)) {}

		Decision update(const std::vector<Detection> &dets, double now)
		{
			expire(now);
			Cand best[3]; // [0] = CHANGE, [1] = STOP, [2] = INTERSECTION
			for (const auto &d : dets)
			{
				// Filter out detections with low confidence
				if (!d.valid || d.score < cfg_.min_score || d.n_pts < cfg_.min_points)
					continue;

				// Filter out detections based on distance
				if (!std::isfinite(d.distance) || d.distance < cfg_.min_distance)
					continue;

				// Filter out clutter that is physically way outside our lane width (using local base frame Y)
				if (std::abs(d.local_y) > cfg_.max_lateral_offset)
					continue;

				const auto it = cfg_.classes.find(d.cls);
				// Detection doesnt have an action
				if (it == cfg_.classes.end() || it->second.action == Action::NONE)
					continue;

				const ClassRule &r = it->second;

				if (r.mark_stop_sign && d.distance <= r.trigger_distance) 
					stop_sign_time_ = now;

				if (r.action == Action::NONE)
					continue;

				if (d.distance > r.trigger_distance)
					continue; // not close enough yet

				if ((r.action == Action::CHANGE || r.action == Action::INTERSECTION) && handled(d, now))
					continue; // already dealt with

				Cand &c = best[slot(r.action)];
				if (!c.valid || d.distance < c.det.distance)
				{
					c.valid = true;
					c.det = d;
				}
			}

			bool active[3] = {false, false, false};
			const int confirm[3] = {cfg_.confirm_change, cfg_.confirm_stop, cfg_.confirm_intersection};
			const double hold[3] = {cfg_.hold_time, cfg_.hold_time, cfg_.intersection_hold};
			for (int i = 0; i < 3; ++i)
			{
				hits_[i] = best[i].valid ? hits_[i] + 1 : 0;
				if (best[i].valid && hits_[i] >= confirm[i])
				{
					hold_until_[i] = now + hold[i];
					last_det_[i] = best[i].det;
				}
				active[i] = (best[i].valid && hits_[i] >= confirm[i]) || now < hold_until_[i];
			}

			Decision out;
			if (active[1])
			{
				out = make(Action::STOP, last_det_[1]);
			}
			else if (active[2]) 
			{
				out = make(Action::INTERSECTION, last_det_[2]);
				out.needs_stop = (now - stop_sign_time_) < cfg_.stop_sign_ttl;
			}
			else if (active[0])
			{
				out = make(Action::CHANGE, last_det_[0]);
			}
			return out;
		}

		// Call when a action for exactly this decision (the one the goal was sent for) succeeded.
		void mark_handled(const Decision &d, double now)
		{
			if (d.action != Action::CHANGE && d.action != Action::INTERSECTION)
				return;

			Handled h;
			h.id = d.obstacle_id;
			h.cls = d.cls;
			h.distance = d.distance;
			h.x = d.x;
			h.y = d.y;
			h.last_seen = now;
			handled_.push_back(h);

			const int i = slot(d.action);
			hold_until_[i] = 0.0;
			hits_[i] = 0;
			if (d.action == Action::INTERSECTION) 
				stop_sign_time_ = -1e9; // Sign has been used up
		}

	private:
		struct Cand
		{
			bool valid = false;
			Detection det;
		};
		struct Handled
		{
			long id = -1;
			std::string cls;
			double distance = 0.0;
			double x = 0.0;
			double y = 0.0;
			double last_seen = 0.0;
		};

		static Decision make(Action a, const Detection &d)
		{
			Decision o;
			o.action = a;
			o.obstacle_id = d.id;
			o.distance = d.distance;
			o.cls = d.cls;
			o.x = d.x;
			o.y = d.y;
			return o;
		}

		bool handled(const Detection &d, double now)
		{
			for (auto &h : handled_)
			{
				// 1. Check matching real ID if available
				if (d.id >= 0 && h.id >= 0 && h.id == d.id)
				{
					h.last_seen = now;
					return true;
				}

				// 2. Fixed World-Space Exclusion (1.8m around physical obstacle location)
				double dx = d.x - h.x;
				double dy = d.y - h.y;
				double spatial_dist = std::sqrt(dx * dx + dy * dy);

				if (h.cls == d.cls && spatial_dist <= 1.8)
				{
					h.last_seen = now;
					return true;
				}
			}
			return false;
		}

		void expire(double now)
		{
			handled_.erase(std::remove_if(handled_.begin(), handled_.end(),
										  [&](const Handled &h)
										  {
											  return now - h.last_seen > (h.id >= 0 ? cfg_.handled_ttl : cfg_.pseudo_ttl);
										  }),
						   handled_.end());
		}

		DecisionConfig cfg_;
		int hits_[3] = {0, 0, 0};
		double hold_until_[3] = {0.0, 0.0, 0.0};
		Detection last_det_[3];
		std::vector<Handled> handled_;
		double stop_sign_time_ = -1e9;
	};

}
