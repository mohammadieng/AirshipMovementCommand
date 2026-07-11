# AirshipMovementCommand
What if you could build an autonomous aerial observer for just a few dollars — no GPS, no LiDAR, just a party balloon and a camera? 🎈📷

The problem:
Getting a stable, bird’s-eye view of a site (farm, construction, field survey) usually means expensive drones or complex sensor setups. What if you could hover in place using only a single downward-facing camera, keeping it cheap, simple, and tether-free?

How I solved it:
I built an autonomous blimp that holds position using pure visual odometry — no external positioning needed. Here’s the short technical recipe:

- Perception: ORB feature detection (lightweight, real-time) + FLANN matching with Lowe’s ratio test to track the ground across frames.
- Motion estimation: A homography matrix captures how the ground plane shifts; from that, I extract drift.
- Noise handling: A Kalman filter fuses prediction (velocity model) and measurements. When features fail (bland surfaces), it falls back on prediction alone — no freakouts.
- Smoothing & control: A moving average over the last 30 positions reduces jitter, then a simple error signal (dx, dy) feeds a PID controller that commands left/right/up/down to keep the blimp over the target.

The result? The blimp builds a “mental map” of its drift and corrects itself in real time, using just one camera. 🧠

What’s next:
When the ground has zero texture — water, sand, snow — traditional features break. That’s where optical flow and tiny deep learning models could take over. I’m exploring lightweight neural trackers that can run on board, keeping the same spirit of minimal hardware. If you’re into weird robots, visual SLAM, or DIY autonomy, I’d love to hear your ideas.

Give the repo a star, try it on your own blimp (or drone!), and let me know how robust you can make it. 🚀
