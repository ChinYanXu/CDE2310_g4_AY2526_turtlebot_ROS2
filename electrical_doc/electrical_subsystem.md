# Electrical Subsystem
Our Electrical Subsystem consists of the following: 
1. System Overview
2. Key Components
3. Details on the Mechanism
   - Station A
   - Station B
4. Power Calculations
5. Schematics 

## System Overview
To fulfill the mission requirements, we built a dual-gate mechanism to deposit balls into the targets one at a time. Before the run starts, six balls are loaded onto the ramp behind the inner gate. When the robot docks, the inner gate opens to let exactly one ball roll into the middle holding chamber between the outer and inner gate. Then, the outer gate opens to drop that single ball into the tin can.
Two SG90 servos act as the physical gates and are driven by PWM signals directly from the Raspberry Pi's GPIO pins. The inner gate separates the ramp from the holding chamber and the outer gate separates the holding chamber from the tin can. \
**Ball path:**  
```
Ramp → [Inner Gate] → Holding Chamber → [Outer Gate] → Tin Can
```

## Key Components 
| Payload Components | Units|
| :--- | :--- |
| SG90 Servo | x2 |
| USB Camera | x2 |

| Existing Components | Units|
| :--- | :--- |
| LiDAR | x1 |
| Dynamixel Motors | x2 |
| LiPO Battery (19.98Wh) | x1 |
| Raspberry Pi 4 | x1 |
| OpenCR | x1  |


## Details on the Mechanism
The payload delivery is controlled by a dedicated ROS 2 node (`payload_delivery_node`) that manages the two servos using a non-blocking queue system, so servo movements can execute without stalling the rest of the system. Whenever the robot starts a run, the servos are initialized to their horizontal (closed) positions to block the balls. The delivery logic is split into two distinct mission protocols:

### Station A 
Our group's specific delivery timing sequence is 6-4. Station A logic executes a fully automated, time-based release sequence to drop three balls. After the robot docks, a `START_A` command is sent over the `/station_cmd` topic to trigger the payload sequence.

#### Sequence: 
1. **First Ball**:
   Inner gate opens and closes to load the first ball into the chamber, immediately followed by the outer gate opening and closing to drop it. 
2.  **Second Ball**:
The system waits for 5 seconds, then repeats the sequence for the second ball. This 5-second delay, combined with the 1-second physical drop time, satisfies the 6-second interval requirement.
3. **Third Ball**:
The system waits for another 3 seconds and then repeats the sequence for the third ball. This 3-second delay, combined with the 1-second physical drop time, satisfies the 4-second interval requirement.
4. **Mission Complete**:
Once finished, the node publishes a `FINISH_A` message to the `/mission_complete` topic so the robot can resume movement.

### Station B 
Station B logic uses the Aruco Marker detection to drop the balls. When the robot docks, a `START_B` command activates the system but the balls do not drop immediately. Instead, each ball is dispensed one at a time whenever the USB camera detects an Aruco Marker (Tag ID 3) placed inside the moving tin can. 

To prevent the system from double-firing for the same visual frame, there is a 2 second cooldown between triggers. Delivering all 3 balls requires 4 separate Aruco marker detections (passes).

#### Sequence: 
1. **First detection**: Inner gate opens to load the first ball into the chamber.
2. **Second detection**: Outer gate opens to drop the first ball. Inner gate immediately loads the second ball into the chamber.
3. **Third detection**: The outer gate opens to drop the second ball into tin can and inner gate immediately loads the third ball into chamber.
4. **Fourth detection**: The outer gate drops the third ball into the tin. The node disarms the sequence and publishes a `FINISH_B` message so that the robot can resume movement.


## Power Calculations 
To ensure the robot can operate reliably during the full mission, we calculated the total power draw of the base system and all attached peripherals.

### Base Robot Power Consumption
| Stages | Power Calculation |
| :--- | :--- |
| **Initial Boot Up** | 11.1V x 800mA = **8.88W** |
| **Idle** | 11.1V x 550mA = **6.11W** |
| **During Operation** | 11.1V x 625mA = **6.94W** |

### Peripheral Power Consumption
*Note: Table above reflects the full Turtlebot base system, which already includes the Raspberry Pi and LiDAR. These are therefore not counted again in the calculations below.*

| Components | Power (Per Unit) | Quantity | Total Power |
| :--- | :--- | :---: | :--- |
| **SG90 Servo** | Swinging: 5V x 100mA = 0.5W<br>Stall: 5V x 360mA = 1.8W | x2 | Swinging: 0.5W x 2 = **1W**<br>Stall: 1.8W x 2 = **3.6W** |
| **USB Camera** | 5V x 350mA = 1.75W | x2 | 1.75W x 2 = **3.5W** |
| **Total Minimum Power Consumption (servos swinging)** |-|-| **4.5W** |
| **Total Maximum Power Consumption (servos stalling)** |-|-| **7.1W** |


### System Power Draw
| Scenario | Base Robot | Peripherals | Total Draw |
| :--- | :--- | :--- | :--- |
| **Minimum** | 6.11W *(Idle)* | 4.5W | **10.61W** |
| **Normal Operation under Minimum Load** | 6.94W | 4.5W | **11.44W** |
| **Normal Operation under Maximum Load** | 6.94W | 7.1W | **14.04W** |
| **Maximum** | 8.88W *(Boot)* | 7.1W | **15.98W** |

**Safe Usage:** Maximum power draw does not exceed capacity

### Estimated Runtime
*Runtime (h) = Battery Capacity (Wh) ÷ Total Power (W)*

| Scenario | Calculation | Estimated Runtime |
| :--- | :--- | :--- |
| **Minimum** | 19.98Wh ÷ 10.61W | **~113.0min** |
| **Normal Operation under Minimum Load** | 19.98Wh ÷ 11.44W | **~104.8min** |
| **Normal Operation under Maximum Load** | 19.98Wh ÷ 14.04W | **~85.4min** |
| **Maximum** | 19.98Wh ÷ 15.98W | **~75.0min** |

The calculated runtimes for all scenarios are significantly more than required mission duration.

## Schematics
### Electrical Conceptual Diagram
<img width="1781" height="1156" alt="image" src="https://github.com/user-attachments/assets/203e70d0-bf2c-4f29-aab1-442d805ce6be" />

### Electrical Wiring 
**OpenCR**:
<img width="2128" height="1644" alt="image" src="https://github.com/user-attachments/assets/42a9c008-330a-4d79-a182-ca96603f030d" />

**Raspberry Pi**:
<img width="1805" height="1745" alt="image" src="https://github.com/user-attachments/assets/32616024-529a-4825-8c32-1134a4858e68" />
The servos are connected to 5V and GND pins of the Raspberry Pi, with GPIO pins 18 and 12 configured as control signals. The cameras are connected through the Raspberry Pi’s USB ports.
