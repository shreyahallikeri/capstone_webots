from controller import Robot, Supervisor  # type: ignore
from typing import Tuple, List
import numpy as np


class Mavic(Supervisor):
    def __init__(self, robot: Robot):
        super().__init__()
        # Initialize the Robot
        self.drone = robot
        self.timestep = int(self.drone.getBasicTimeStep())

        # Enable Camera
        self.camera = self.drone.getDevice("camera")
        self.camera.enable(self.timestep)

        # Enable IMU (Inertial Measurement Unit)
        self.imu = self.drone.getDevice("inertial unit")
        self.imu.enable(self.timestep)

        # Enable Gyroscope
        self.gyro = self.drone.getDevice("gyro")
        self.gyro.enable(self.timestep)

        # Enable GPS
        self.gps = self.drone.getDevice("gps")
        self.gps.enable(self.timestep)

        # Enable Range Finder
        self.range_finder = self.drone.getDevice("my_range_finder")
        self.range_finder.enable(self.timestep)

        # Initialize Propellers (Motors)
        motor_names = [
            "front left propeller", "front right propeller", 
            "rear left propeller", "rear right propeller"
        ]
        
        self.motors = [self.drone.getDevice(motor_name) for motor_name in motor_names]
        
        for motor in self.motors:
            motor.setPosition(float('inf'))  # Set motors to velocity control mode
            motor.setVelocity(0.0)  # Initialize motor velocity to zero

    def get_imu_values(self) -> Tuple[float, float, float]:
        """Returns the roll, pitch, and yaw values from the IMU."""
        return self.imu.getRollPitchYaw()

    def get_gps_values(self) -> Tuple[float, float, float]:
        """Returns the x, y, and z coordinates from the GPS."""
        return self.gps.getValues()
        
    def get_depth_data(self):
        """Retrieve the depth data from the range finder."""
        range_image = self.range_finder.getRangeImage()  # Get the 2D range image
        return np.array(range_image)

    def get_gyro_values(self) -> Tuple[float, float, float]:
        """Returns the angular velocity around x, y, and z axes from the gyro."""
        return self.gyro.getValues()

    def get_time(self) -> float:
        """Returns the current simulation time."""
        return self.drone.getTime()

    def set_rotor_speed(self, speeds: Tuple[float, float, float, float]) -> None:
        """Sets the velocity for each rotor."""
        for motor, speed in zip(self.motors, speeds):
            motor.setVelocity(speed)

    def get_image(self) -> List[List[List[int]]]:
        return self.camera.getImage()
    
    def step_robot(self) -> int:
        """Increments the simulation step."""
        return self.drone.step(self.timestep)
    
    def reset(self) -> None:
        """Reset the simulation with complete reinitialization of components."""
        # Reset simulation and physics
        self.simulationReset()
        self.simulationResetPhysics()
        
        self.drone.step(self.timestep)
        
        # Reinitialize sensors by disabling and enabling them to reset sensor states
        self.camera.disable()
        self.imu.disable()
        self.gyro.disable()
        self.gps.disable()
        
        self.camera.enable(self.timestep)
        self.imu.enable(self.timestep)
        self.gyro.enable(self.timestep)
        self.gps.enable(self.timestep)
        
        # Reinitialize motor settings and reset motor velocities
        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)
        
        # Add a small delay step to fully apply the reset before resuming control
        for _ in range(5):
            self.drone.step(self.timestep)

