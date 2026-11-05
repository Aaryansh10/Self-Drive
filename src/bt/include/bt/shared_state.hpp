// Thread-safe bridge between the detection subscriber and the BT nodes.
#pragma once
#include <functional>
#include <mutex>
#include <utility>
#include <vector>
#include <deque>
#include <cstdint>

#include "bt/decision.hpp"

namespace bt
{

	class SharedState
	{
	public:
		// stale_timeout: if no detection list arrived for this long, the decision falls back to NONE
		SharedState(DecisionConfig cfg, double stale_timeout, std::function<double()> clock)
			: decider_(std::move(cfg)), stale_(stale_timeout), clock_(std::move(clock)) {}

		double now() const { return clock_(); }

		void on_detections(const std::vector<Detection> &dets)
		{
			const double t = clock_();
			std::lock_guard<std::mutex> l(m_);
			decision_ = decider_.update(dets, t);
			t_msg_ = t;
			have_ = true;
		}

		Decision current()
		{
			const double t = clock_();
			std::lock_guard<std::mutex> l(m_);
			if (!have_ || t - t_msg_ > stale_)
				return Decision{};
			return decision_;
		}

		// a lane change for this exact decision succeeded
		void mark_handled(const Decision &d)
		{
			const double t = clock_();
			std::lock_guard<std::mutex> l(m_);
			decider_.mark_handled(d, t);
			if (decision_.action == Action::CHANGE || decision_.action == Action::INTERSECTION)
				decision_ = Decision{};
		}

		void set_turns(const std::vector<uint8_t>& t)
		{
			std::lock_guard<std::mutex> l(m_);
			turns_.assign(t.begin(), t.end());
		}

		bool peek_turn(uint8_t& turn)
		{
			std::lock_guard<std::mutex> l(m_);
			if (turns_.empty())
				return false;
			turn = turns_.front();
			return true;
		}

		void pop_turn()
		{
			std::lock_guard<std::mutex> l(m_);
			if (!turns_.empty())
				turns_.pop_front();
		}

	private:
		std::mutex m_;
		Decider decider_;
		Decision decision_;
		double stale_;
		double t_msg_ = 0.0;
		bool have_ = false;
		std::function<double()> clock_;
		std::deque<uint8_t> turns_;
	};

} // namespace bt