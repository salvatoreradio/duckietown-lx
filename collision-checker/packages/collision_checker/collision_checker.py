import itertools
import random
import numpy as np
from typing import List

from aido_schemas import Context, FriendlyPose
from dt_protocols import (
    Circle,
    CollisionCheckQuery,
    CollisionCheckResult,
    MapDefinition,
    PlacedPrimitive,
    Rectangle,
    Primitive
)

__all__ = ["CollisionChecker"]


class CollisionChecker:
    params: MapDefinition

    def init(self, context: Context):
        context.info("init()")

    def on_received_set_params(self, context: Context, data: MapDefinition):
        context.info("initialized")
        self.params = data

    def on_received_query(self, context: Context, data: CollisionCheckQuery):
        collided = check_collision(
            environment=self.params.environment, robot_body=self.params.body, robot_pose=data.pose
        )
        result = CollisionCheckResult(collided)
        context.write("response", result)


def check_collision(
    environment: List[PlacedPrimitive], robot_body: List[PlacedPrimitive], robot_pose: FriendlyPose
) -> bool:
    # This is just some code to get you started, but you don't have to follow it exactly
    robot_part: PlacedPrimitive
    rototranslated_robot: List[PlacedPrimitive] = []
    # TODO you can start by rototranslating the robot_body by the robot_pose
    for robot_part in robot_body:
        robot_part_pose = robot_part.pose
        robot_part_primitive = robot_part.primitive
        if isinstance(robot_part_primitive, Rectangle):
            robot_part_theta_rad = np.radians(robot_part_pose.theta_deg)
            robot_pose_theta_rad = np.radians(robot_pose.theta_deg)
            
            x = robot_part_pose.x + (np.cos(robot_part_theta_rad) * robot_pose.x - np.sin(robot_part_theta_rad) * robot_pose.y)
            y = robot_part_pose.y + (np.sin(robot_part_theta_rad) * robot_pose.x + np.cos(robot_part_theta_rad) * robot_pose.y)
        
            theta_deg = robot_part_pose.theta_deg + robot_pose.theta_deg
            
            new_robot_part_pose = FriendlyPose(x, y, theta_deg)
            
            rotated_vertices = get_vertices(robot_part_pose, robot_part_primitive)
            
            xmin, ymin = np.min(rotated_vertices, axis=0)
            xmax, ymax = np.max(rotated_vertices, axis=0)
            
            new_robot_part_primitive = Rectangle(xmin, ymin, xmax, ymax)
            
            new_robot = PlacedPrimitive(new_robot_part_pose, new_robot_part_primitive)
            
        elif isinstance(robot_part_primitive, Circle):
            x = robot_part_pose.x + robot_part.x
            y = robot_part_pose.y + robot_part.y
            
            new_robot_part_pose = FriendlyPose(x, y, robot_part_pose.theta_deg)
            new_robot_part_primitive = Circle(robot_part_primitive.radius)
        
        rototranslated_robot.append(new_robot)
    

    # Then, call check_collision_list to see if the robot collides with the environment
    collided = check_collision_list(rototranslated_robot, environment)

    # TODO return the status of the collision
    return collided


def check_collision_list(
    rototranslated_robot: List[PlacedPrimitive], environment: List[PlacedPrimitive]
) -> bool:
    # This is just some code to get you started, but you don't have to follow it exactly
    for robot, envObject in itertools.product(rototranslated_robot, environment):
        if check_collision_shape(robot, envObject):
            return True

    return False


def check_collision_shape(a: PlacedPrimitive, b: PlacedPrimitive) -> bool:
    # This is just some code to get you started, but you don't have to follow it exactly
    robot_primitive = a.primitive
    robot_pose = a.pose
    env_primitive = b.primitive
    env_pose = b.pose
    center_distance = np.linalg.norm(np.array((env_pose.x, env_pose.y)) - np.array(robot_pose.x, robot_pose.y))
    
    # TODO check if the two primitives are colliding
    if isinstance(a.primitive, Circle) and isinstance(b.primitive, Circle):
        if center_distance > (robot_primitive.radius + env_primitive.radius):
            return False
        
        return True
        
    if isinstance(a.primitive, Rectangle) and isinstance(b.primitive, Circle):
        env_center = (env_pose.x, env_pose.y)
        robot_center = (robot_pose.x, robot_pose.y)
        
        theta = np.radians(robot_pose.theta_deg)
        dx = env_center.x - robot_center.x
        dy = env_center.y - robot_center.y
        
        x_local = dx * np.cos(theta) - dy * np.sin(theta)
        y_local = dx * np.sin(theta) + dy * np.cos(theta)
        
        nearest_x = max(robot_primitive.xmin, min(x_local, robot_primitive.xmax))
        nearest_y = max(robot_primitive.ymin, min(y_local, robot_primitive.ymax))
        
        distance = ((nearest_x - x_local) ** 2 + (nearest_y - y_local) ** 2) ** 0.5
        
        if distance > env_primitive.radius:
            return False
        
        return True
        
        
        
    if isinstance(a.primitive, Rectangle) and isinstance(b.primitive, Rectangle):
        
        robot_vertices = get_vertices(robot_pose, robot_primitive)
        env_vertices = get_vertices(env_pose, env_primitive)
        
        edges1 = [robot_vertices[i] - robot_vertices[i - 1] for i in range(len(robot_vertices))]
        edges2 = [env_vertices[i] - env_vertices[i - 1] for i in range(len(env_vertices))]
        axes = [np.array([-e[1], e[0]]) for e in edges1 + edges2]
        
        for ax in axes:
            robot_projection = [np.dot(v, ax) for v in robot_vertices]
            env_projection = [np.dot(v, ax) for v in env_vertices]
            
            if max(robot_projection) < min(env_projection) or max(env_projection) < min(robot_projection):  # C'è separazione
                return False
        
        return True


def get_vertices(pose: FriendlyPose, primitive: Primitive):
    
    theta = np.radians(pose.theta_deg)
    
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    local_vertices = np.array([
        [primitive.xmin, primitive.ymin],
        [primitive.xmin, primitive.ymax],
        [primitive.xmax, primitive.ymax],
        [primitive.xmax, primitive.ymin]
    ])
    
    return np.dot(local_vertices, R.T) + np.array([pose.x, pose.y])