# start by importing some things we will need
import numpy as np
from math import floor, sqrt
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import multivariate_normal


# Now let's define the prior function. In this case we choose
# to initialize the historgram based on a Gaussian distribution around [0,0]
def histogram_prior(belief, grid_spec, mean_0, cov_0):
    pos = np.empty(belief.shape + (2,))
    pos[:, :, 0] = grid_spec["d"]
    pos[:, :, 1] = grid_spec["phi"]
    RV = multivariate_normal(mean_0, cov_0)
    belief = RV.pdf(pos)
    return belief

def wrap_angle(angle, phi_min, phi_max):
    phi_range = phi_max - phi_min
    while angle < phi_min:
        angle += phi_range
    while angle > phi_max:
        angle -= phi_range
    return angle

# Now let's define the predict function


def histogram_predict(belief, left_encoder_ticks, right_encoder_ticks, grid_spec, robot_spec, cov_mask):
    belief_in = belief

    # TODO calculate v and w from ticks using kinematics.
    #  You will need  some parameters in the `robot_spec` defined above
    encoder_resolution = robot_spec["encoder_resolution"]
    wheel_radius = robot_spec["wheel_radius"]
    wheel_baseline = robot_spec["wheel_baseline"]
    d_left = (2 * np.pi * wheel_radius * left_encoder_ticks) / encoder_resolution
    d_right = (2 * np.pi * wheel_radius * right_encoder_ticks) / encoder_resolution
    
    # You may find the following code useful to find the current best heading estimate:
    maxids = np.unravel_index(belief_in.argmax(), belief_in.shape)
    phi_max = grid_spec['phi_min'] + (maxids[1] + 0.5) * grid_spec['delta_phi']

    v = ((d_left + d_right) / 2) * np.cos(phi_max)  # replace this with a function that uses the encoder
    w = (d_right - d_left) / wheel_baseline  # replace this with a function that uses the encoder

    # propagate each centroid
    d_t = grid_spec["d"] + v
    phi_t = grid_spec["phi"] + w
    phi_t = wrap_angle(phi_t, grid_spec["phi_min"], grid_spec["phi_max"])

    p_belief = np.zeros(belief.shape)

    # Accumulate the mass for each cell as a result of the propagation step
    for i in range(belief.shape[0]):
        for j in range(belief.shape[1]):
            # If belief[i,j] there was no mass to move in the first place
            if belief[i, j] > 0:
                # Now check that the centroid of the cell wasn't propagated out of the allowable range
                if (
                    d_t[i, j] > grid_spec["d_max"]
                    or d_t[i, j] < grid_spec["d_min"]
                    or phi_t[i, j] < grid_spec["phi_min"]
                    or phi_t[i, j] > grid_spec["phi_max"]
                ):
                    continue

                # TODO Now find the cell where the new mass should be added
                i_new = int((d_t[i, j] - grid_spec["d_min"]) / (grid_spec["d_max"] - grid_spec["d_min"]) * belief.shape[0])
                phi_t[i, j] = wrap_angle(phi_t[i, j], grid_spec["phi_min"], grid_spec["phi_max"])
                j_new = int((phi_t[i, j] - grid_spec["phi_min"]) / (grid_spec["phi_max"] - grid_spec["phi_min"]) * belief.shape[1])

                # Assicurati che gli indici siano validi
                i_new = min(max(i_new, 0), belief.shape[0] - 1)  # replace with something that accounts for the movement of the robot
                j_new = min(max(j_new, 0), belief.shape[1] - 1)  # replace with something that accounts for the movement of the robot

                p_belief[i_new, j_new] += belief[i, j]

    # Finally we are going to add some "noise" according to the process model noise
    # This is implemented as a Gaussian blur
    s_belief = np.zeros(belief.shape)
    gaussian_filter(p_belief, cov_mask, output=s_belief, mode="constant")

    if np.sum(s_belief) == 0:
        return belief_in
    belief = s_belief / np.sum(s_belief)
    return belief


# We will start by doing a little bit of processing on the segments to remove anything that is
# behing the robot (why would it be behind?) or a color not equal to yellow or white


def prepare_segments(segments, grid_spec):
    filtered_segments = []
    for segment in segments:

        # we don't care about RED ones for now
        if segment.color != segment.WHITE and segment.color != segment.YELLOW:
            continue
        # filter out any segments that are behind us
        if segment.points[0].x < 0 or segment.points[1].x < 0:
            continue

        point_range = getSegmentDistance(segment)
        if grid_spec["range_est"] > point_range > 0:
            filtered_segments.append(segment)
    return filtered_segments


def generate_vote(segment, road_spec):
    p1 = np.array([segment.points[0].x, segment.points[0].y])
    p2 = np.array([segment.points[1].x, segment.points[1].y])
    t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)
    n_hat = np.array([-t_hat[1], t_hat[0]])

    d1 = np.inner(n_hat, p1)
    d2 = np.inner(n_hat, p2)
    l1 = np.inner(t_hat, p1)
    l2 = np.inner(t_hat, p2)
    if l1 < 0:
        l1 = -l1
    if l2 < 0:
        l2 = -l2

    l_i = (l1 + l2) / 2
    d_i = (d1 + d2) / 2
    phi_i = np.arcsin(t_hat[1])
    if segment.color == segment.WHITE:  # right lane is white
        if p1[0] > p2[0]:  # right edge of white lane
            d_i -= road_spec["linewidth_white"]
        else:  # left edge of white lane
            d_i -= road_spec["linewidth_white"]
            d_i = road_spec["lanewidth"] * 2 + road_spec["linewidth_yellow"] - d_i
            phi_i = -phi_i
        d_i -= road_spec["lanewidth"]/2

    elif segment.color == segment.YELLOW:  # left lane is yellow
        if p2[0] > p1[0]:  # left edge of yellow lane
            d_i -= road_spec["linewidth_yellow"]
            d_i = road_spec["lanewidth"]/2 - d_i
            phi_i = -phi_i
        else:  # right edge of yellow lane
            d_i += road_spec["linewidth_yellow"]
            d_i -= road_spec["lanewidth"]/2

    return d_i, phi_i


def generate_measurement_likelihood(segments, road_spec, grid_spec):
    # initialize measurement likelihood to all zeros
    measurement_likelihood = np.zeros(grid_spec["d"].shape)

    num_cells_d = grid_spec['d'].shape[0]
    num_cells_phi = grid_spec['phi'].shape[1]

    for segment in segments:
        d_i, phi_i = generate_vote(segment, road_spec)
        phi_i = wrap_angle(phi_i, grid_spec["phi_min"], grid_spec["phi_max"])

        # if the vote lands outside of the histogram discard it
        if (
            d_i > grid_spec["d_max"]
            or d_i < grid_spec["d_min"]
            or phi_i < grid_spec["phi_min"]
            or phi_i > grid_spec["phi_max"]
        ):
            continue

        # So now we have d_i and phi_i which correspond to the estimate of the distance
        # from the center and the angle relative to the center generated by the single
        # segment under consideration
        # TODO find the cell index that corresponds to the measurement d_i, phi_i
        i = int((d_i - grid_spec['d_min']) / (grid_spec['d_max'] - grid_spec['d_min']) * num_cells_d)
        j = int((phi_i - grid_spec['phi_min']) / (grid_spec['phi_max'] - grid_spec['phi_min']) * num_cells_phi)

        i = min(max(i, 0), num_cells_d - 1)  # replace this
        j = min(max(j, 0), num_cells_phi - 1)  # replace this

        # Add one vote to that cell
        measurement_likelihood[i, j] += 1

    if np.linalg.norm(measurement_likelihood) == 0:
        return None
    measurement_likelihood /= np.sum(measurement_likelihood)
    return measurement_likelihood


def histogram_update(belief, segments, road_spec, grid_spec):
    # prepare the segments for each belief array
    segmentsArray = prepare_segments(segments, grid_spec)
    # generate all belief arrays

    measurement_likelihood = generate_measurement_likelihood(segmentsArray, road_spec, grid_spec)

    if measurement_likelihood is not None:
        # TODO: combine the prior belief and the measurement likelihood to get the posterior belief
        # Don't forget that you may need to normalize to ensure that the output is valid
        # probability distribution

        posterior_belief = belief * measurement_likelihood # replace this with something that combines the belief and the measurement_likelihood
        if np.sum(posterior_belief) > 0:
            posterior_belief /= np.sum(posterior_belief)
        belief = posterior_belief
        
    return measurement_likelihood, belief

def getSegmentDistance(segment):
    x_c = (segment.points[0].x + segment.points[1].x) / 2
    y_c = (segment.points[0].y + segment.points[1].y) / 2
    return sqrt(x_c**2 + y_c**2)