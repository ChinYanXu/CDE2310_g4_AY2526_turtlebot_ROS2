# TurtleBot Payload mechanical documentation

This document provides detailed instructions for printing, assembling, and integrating the payload onto your TurtleBot.

---

## Components

### TurtleBot3 Subcomponents
- LiDAR
- RaspberryPi
- OpenCR 1.0
- 2x Dynamixel motors
- LiPo Battery
- USB2LDS

### Bill of Materials
- Adafruit PCA9685 16-Channel Servo Driver
- Arducam OV5648 5MP USB Camera x2
- SG90 Servos x2
- Hot Glue Gun x1
- Alt Tab PLA Filament 1kg

### Printed Parts
- **Servo Mounts**: Individually printed servo mounts. (Print 1 L and 1 R)
- **Camera Mounts**: Individually printed camera mounts. (Print 1 High and 1 Low)
- **Ball Magazine Ramp**: Individually printed.
- **Ball Holder**: Individually printed.
- **Ball Ramp**: Individually printed.
- **Gate**: 2 pieces individually printed.

### Hardware
- **Extra Waffle Plate**: Extra floor for payload.
- **10x 30mm Standoff MF**
- **5x 15mm Standoff MF**
- **6x M2 Nut**
- **4x M4 Nut**
- **3x M6 Nut**
- **6x M2x16**
- **4x M4x16**
- **3x M6x16**

---

## Assembly Instructions

### Step 1: Preparation
1. **Remove TurtleBot Layers**: Remove the top layer of your TurtleBot.

### Step 2: Installing the Payload floor
1. **Installing the ramps**:
   - Postion the 3 ramps according to the CAD file.
   - Secure the ramps to the waffle plates using M6, M4, and M2 nuts and bolts based on the respective holes.

2. **Assembling the servo gates**:
   - Press fit the servo motors to the respective mounts.
   - Align the gate to the first 3 holes of the servo arm. (The long edge of the gate should be parallel to the plate)
   - Use hot glue to secure.
   - Once the glue sets, align the press fit extrusions of the mount to the corresponding holes and press down to secure.

### Step 3: Reinstalling Components
1. **Reassemble TurtleBot Layers**:
   - Attach the payload floor to the TurtleBot.
   - Combine the following standoffs for each pillar, aligning them with the ones on the present layer.
        - 2x 30mm
        - 1x 15mm
   - Attach the removed LIDAR floor to the top of the TurtleBot and secure with bolts.

### Step 4: Camera installation
1. **Secure Camera to mount**: Use ing M2 nuts and bolts, secure the camera to the mount.
2. **Attaching Camera to TurtleBot**: 
   - Align Camera HIGH to the front of the OpenCR Level and press down to secure press fit extrusions to their respective holes.
   - Align Camera LOW to the front of the battery Level and press down to secure press fit extrusions to their respective holes.
---

## Important Installation Notes

### Wiring and Electronics
- **Camera and Servo Motors**: Wire these components after the payload floor is fully secured, to avoid wiring complications.

### Servo Installation
- Servos as press fit into the 3d printed mounts, masking tape can be used to secure further.

---

## Mechanical and Assembly Recommendations
- Ensure all glued components have fully set before testing the release mechanism.
- Double-check servo alignments and wiring connections to ensure smooth operation.
- Conduct initial launch tests at low servo power settings to verify mechanical stability and integrity.

---

## Design Reasoning
- Using a gated release ramp system relies on gravity, removing force calculations vs other methods.
- The slim footprint of this design allows it to navigate narrow spaces easily.
- Design is easy to replicate and print.
---

## Additional Details
- Ensure all wires are secure

---

For further support or inquiries, please contact CDE2310 Group 4 AY25/26 or refer to the provided project documentation.
```
