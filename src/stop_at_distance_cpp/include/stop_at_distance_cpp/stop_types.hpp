#pragma once

#include <cstdint>
#include <string>
#include <cmath>

namespace stop_at_distance {
    enum class Phase : uint8_t {IDLE = 0, BRAKE = 1, SETTLE = 2, HOLD = 3, ABORT = 4};
    enum class Mode : uint8_t {IDLE = 0, BRAKE = 1, HOLD = 2, CONTROLLED_STOP = 3};
    enum class Outcome : uint8_t {SUCCESS = 0, CANCELED = 1, TIMEOUT = 2, OVERSHOOT = 3, SETTLE_TIMEOUT = 4, INTERNAL_FAULT = 5};

const char * to_string(Phase phase);
const char * to_string(Mode mode);
const char * to_string(Outcome outcome);

Mode phase_mode(Phase phase);

class SuccessChecker {
public:
struct Config {
    double lower_boundary{0.0};
    double upper_boundary{0.0};
    double stop_speed_epsilon{0.05};
    double settle_threshold{0.5};
    double max_settle_time{10.0};
};

enum class Decision {PENDING, COMPLETED, OVERSHOOT, SETTLE_TIMEOUT};

void configure(const Config & config);
void reset();
Decision step(double time, double distance, double velocity);

private:
Config eval_config;
double start_time{-1.0};
double settle_start_time{-1.0};       
};

}