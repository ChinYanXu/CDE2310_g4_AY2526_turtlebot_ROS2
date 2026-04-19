import collections
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ros2_aruco_interfaces.msg import ArucoMarkers
import RPi.GPIO as GPIO

class PayloadDeliveryNode(Node):
    def __init__(self):
        super().__init__('payload_delivery_node')
        self.outer_pin, self.inner_pin = 18, 12 
        self.out_open,  self.out_close  = -10,  85
        self.in_open,   self.in_close   =  130, 30
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup([self.outer_pin, self.inner_pin], GPIO.OUT)
        self._pwm_outer = GPIO.PWM(self.outer_pin, 50)
        self._pwm_inner = GPIO.PWM(self.inner_pin, 50)     
        self._pwm_outer.start(0)
        self._pwm_inner.start(0)
        self._inner_close()
        self._outer_close()
        self.a_complete        = False
        self.b_complete        = False
        self.b_armed           = False  
        self.is_mission_active = False  
        self.ball_count_b        = 0
        self.last_b_trigger_time = 0.0
        self.b_cooldown          = 2  
        self._step_queue = collections.deque()
        self._step_timer = None 
        self.create_subscription(String, '/station_cmd', self._station_cmd_cb, 10) 
        self.create_subscription(ArucoMarkers, '/cam_right/aruco_markers', self.aruco_callback, 10)
        self.mission_complete_pub = self.create_publisher(String, '/mission_complete', 10)
        self.get_logger().info('Payload Node Active. Resetting gates to Horizontal...')

    def _station_cmd_cb(self, msg):
        if msg.data == 'START_A' and not self.a_complete and not self.is_mission_active:
            self.run_station_a()
        elif msg.data == 'START_B' and not self.b_complete:
            self.b_armed = True
            self.get_logger().info("Station B ARMED. Resetting for Pass 1.")

    def aruco_callback(self, msg: ArucoMarkers):
        if not self.b_armed or self.b_complete:
            return
        for marker_id in msg.marker_ids:
            if marker_id == 2:
                if self.is_mission_active:
                    self.get_logger().info("ID 3 seen, but servos are currently busy")
                    break
                now = time.time()
                if (now - self.last_b_trigger_time) > self.b_cooldown:
                    self.last_b_trigger_time = now
                    self.get_logger().info("ID 3 TRIGGER SUCCESSFUL")
                    self.run_station_b_step()
                else: 
                    self.get_logger().info("ID 3 seen, but still in cooldown")
                break

    def _enqueue(self, fn, delay: float):
        self._step_queue.append((fn, delay))
        if self._step_timer is None:
            self._run_next_step()

    def _run_next_step(self):
        if not self._step_queue:
            self._step_timer = None
            self.is_mission_active = False
            return
        fn, delay = self._step_queue.popleft()
        self._step_timer = self.create_timer(delay, lambda: self._fire_step(fn))

    def _fire_step(self, fn):
        self.destroy_timer(self._step_timer)
        self._step_timer = None
        fn()
        self._run_next_step()

    def _inner_open(self):
        self.get_logger().info("Servo: inner OPEN")
        duty = 2.5 + (self.in_open / 180.0) * 10.0
        self._pwm_inner.ChangeDutyCycle(duty)
        self._arm_pwm_stop(self._pwm_inner, 0.4) 

    def _inner_close(self):
        self.get_logger().info("Servo: inner CLOSE")
        duty = 2.5 + (self.in_close / 180.0) * 10.0
        self._pwm_inner.ChangeDutyCycle(duty)
        self._arm_pwm_stop(self._pwm_inner, 0.4)

    def _outer_open(self):
        self.get_logger().info("Servo: outer OPEN")
        duty = 2.5 + (self.out_open / 180.0) * 10.0
        self._pwm_outer.ChangeDutyCycle(duty)
        self._arm_pwm_stop(self._pwm_outer, 0.4)

    def _outer_close(self):
        self.get_logger().info("Servo: outer CLOSE")
        duty = 2.5 + (self.out_close / 180.0) * 10.0
        self._pwm_outer.ChangeDutyCycle(duty)
        self._arm_pwm_stop(self._pwm_outer, 0.4)

    def _arm_pwm_stop(self, pwm, delay: float):
        def _stop():
            self.destroy_timer(t)
            pwm.ChangeDutyCycle(0)
        t = self.create_timer(delay, _stop)

    # STATION A 

    def run_station_a(self):
        if self.is_mission_active or self.a_complete:
            return
        self.is_mission_active = True
        self.get_logger().info("Station A: Resetting then Starting...")

        steps = [
            (self._inner_close, 0.0), 
            (self._outer_close, 0.0), 
            (lambda: self.get_logger().info("Reset complete. Starting delivery..."), 0.5), 

            (self._inner_open, 0.0), (self._inner_close, 0.5), 
            (self._outer_open, 0.4), (self._outer_close, 1.0), 

            (lambda: self.get_logger().info("Station A: Ball 2"), 5.0),            #6 second delay
            (self._inner_open, 0.0), (self._inner_close, 0.5),
            (self._outer_open, 0.4), (self._outer_close, 1.0),

            (lambda: self.get_logger().info("Station A: Ball 3"), 3.0),              #4 second delay 
            (self._inner_open, 0.0), (self._inner_close, 0.5),
            (self._outer_open, 0.4), (self._outer_close, 1.0),

            (self._finish_a, 0.5),
        ]
        for fn, delay in steps: self._enqueue(fn, delay)

    # STATION B 

    def run_station_b_step(self):
        self.is_mission_active = True
        self.ball_count_b += 1
        self.get_logger().info(f"Station B: Pass {self.ball_count_b}")

        if self.ball_count_b == 1:
            
            steps = [
                (self._inner_close, 0.0), 
                (self._outer_close, 0.0), 
                (self._inner_open, 0.5), 
                (self._inner_close, 0.5),
            ]
        elif self.ball_count_b in (2, 3):
            steps = [
  
                (self._outer_open, 0.0), 
                (self._outer_close, 0.5),
                (self._inner_open, 0.4),
                (self._inner_close, 0.5),
            ]
        elif self.ball_count_b == 4:
            steps = [
                (self._outer_open, 0.0),
                (self._outer_close, 0.5),
                (self._finish_b, 0.5)
            ]
        else:
            self.is_mission_active = False
            return

        for fn, delay in steps: self._enqueue(fn, delay)

    def _finish_a(self):
        self.a_complete = True
        msg =String()
        msg.data = 'FINISH_A'
        self.mission_complete_pub.publish(msg)
        self.get_logger().info("STATION A FINISHED.")

    def _finish_b(self):
        self.b_complete = True
        self.b_armed = False
        msg = String()
        msg.data = 'FINISH_B'
        self.mission_complete_pub.publish(msg)
        self.get_logger().info("STATION B FINISHED.")

    def destroy_node(self):
        if self._step_timer is not None:
            self.destroy_timer(self._step_timer)
        self._pwm_outer.stop()
        self._pwm_inner.stop()
        GPIO.cleanup()
        super().destroy_node()

def main():
    rclpy.init()
    node = PayloadDeliveryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
