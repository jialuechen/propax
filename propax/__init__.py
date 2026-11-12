"""propax: structure-preserving optimal execution with multi-exponential propagators."""
from .kernels import ExpKernel
from .ops import (ImpactTracker, StreamState, apply_cost_matrix, cost_matrix, fft_impact, impact_path,
                  state_from_history, stream_impact, stream_init, stream_update)
from .qp import QPOptions, QPSolution, solve_qp
from .model import (ImpactModel, Schedule, ScheduleResult, expected_cost, holding_intervals, optimal_cost,
                    optimal_schedule, quadratic_form)
from .clock import BusinessClock, log_linear_gains, log_linear_rate
from .diagnostics import (ManipulationGap, check_model, manipulation_gap, most_profitable_round_trip,
                          round_trip_min_eigenvalue)
from .reference import almgren_chriss_inventory, almgren_chriss_trades
from .portfolio import (FactorCrossImpact, PortfolioResult, PortfolioSchedule, apply_impact, dense_hessian,
                        hessian_apply, linear_term, optimal_portfolio_schedule, portfolio_cost)
from .calibrate import (BootstrapResult, ClockFit, KernelFit, bootstrap_kernel, decay_features, delta_method,
                        fit_kernel, fit_kernel_and_clock, fit_kernel_pooled, kernel_value_covariance,
                        regression_design)
from .simulate import random_order_flow, simulate_prices

__version__ = "0.1.0"
