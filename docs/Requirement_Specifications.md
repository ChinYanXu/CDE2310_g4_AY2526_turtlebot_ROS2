---
title: Requirements Specification
---

# Navigation

- [Back to main README.md](../README.md)

---
### Objectives
We are tasked with simulating goods delivery by a warehouse AMR. The problem can be summarised to a few key objectives:

1) Map out the maze zone
2) Identify the location of the static delivery zone. Navigate to the static delivery zone. Deliver 3 ping pong balls into the static delivery zone without falling out within 10 seconds.
3) Identify the location of the dynamic delivery zone. Navigate to the dynamic delivery zone. Deliver 3 ping pong balls into the dynamic delivery zone.
4) Identify the location of the lift zone. Navigate to the lift zone. Enter the lift, activate the lift. Leave the lift when the lift reaches the second level. 
5) Dispense ping pong ball into the delivery zone (This is a bonus objective)
6) Identify the location of the exit. Navigate to the exit.

### Constraints
These key objectives must be completed within the following set of constraints:

1) 25 minute time limit for setup, mission and cleanup
2) Bot navigates using SLAM, lidar or computer vision
3) Maximum of 6 temporary markers can be used

### Functional Requirements
We make the assumption that we will not be attempting the bonus objective. From the remaining key objectives, we distilled a few functional requirements for our robot:

| Requirement Category | Functional Requirement                                                                                           |
|----------------------|------------------------------------------------------------------------------------------------------------------|
| Navigation           | Our robot needs frontier exploration capabilities                                                                |
| Navigation           | Our robot needs mapping capabilities                                                                             |
| Navigation           | Our robot needs to identify waypoints                                                                            |
| Navigation           | Our robot needs to move to identified waypoints                                                                  |
| Payload delivery     | Our robot needs to detect the distance and position of the delivery zones                                        |
| Payload delivery     | Our robot needs to reliably dispense the ping pong balls into the delivery zones with the correct timing.        |
| Payload delivery     | Our robot needs to reliably predict the movement of the receptacle at the dynamic delivery zone                  |
| Payload delivery     | Our robot needs to reliably dispense the ping pong balls into the delivery zones with the correct timing.        |

--- 