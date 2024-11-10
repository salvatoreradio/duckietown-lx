from typing import Tuple

import numpy as np


def PIDController(
        v_0: float,
        theta_ref: float,
        theta_hat: float,
        prev_e: float,
        prev_int: float,
        delta_t: float
) -> Tuple[float, float, float, float]:
    """
    PID performing heading control.
    Args:
        v_0:        linear Duckiebot speed (given).
        theta_ref:  reference heading pose.
        theta_hat:  the current estiamted theta.
        prev_e:     tracking error at previous iteration.
        prev_int:   previous integral error term.
        delta_t:    time interval since last call.
    Returns:
        v_0:     linear velocity of the Duckiebot
        omega:   angular velocity of the Duckiebot
        e:       current tracking error (automatically becomes prev_e at next iteration).
        e_int:   current integral error (automatically becomes prev_int at next iteration).
    """

    # TODO: these are random values, you have to implement your own PID controller in here
    if (delta_t == 0.0):
        delta_t = 0.07
    
    e = theta_ref - theta_hat
    
    e_int = prev_int + e * delta_t
    # anti-windup - preventing the integral error from growing too much
    e_int = max(min(e_int,2),-2)
    
    e_der = (e - prev_e) / delta_t
    
    
    k_p = 5
    k_i = 0.2
    k_d = 0.1
    
    
    omega = k_p * e + k_i * e_int + k_d * e_der
    
    
    
    # Hint: print for debugging
    print(f"\n\nDelta time : {delta_t} \nE : {np.rad2deg(e)} \nE int : {e_int} \nPrev e : {prev_e} \nTheta hat: {np.rad2deg(theta_hat)} \n")
    # ---
    return v_0, omega, e, e_int
