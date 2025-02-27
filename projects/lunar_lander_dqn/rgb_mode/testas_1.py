import gymnasium
import time
import cv2  # OpenCV for image display
import numpy as np

def main():
    # Create the LunarLander-v3 environment with RGB array rendering
    env = gymnasium.make("LunarLander-v3", render_mode="rgb_array")
    
    # Reset the environment and get the initial observation
    observation, info = env.reset()
    
    done = False
    while not done:
        # Choose a random action from the action space
        action = env.action_space.sample()
        
        # Take a step in the environment
        observation, reward, done, truncated, info = env.step(action)

        # Capture the current frame as a NumPy array
        frame = env.render()
        
        # Convert RGB to BGR (OpenCV format)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Show the frame using OpenCV
        cv2.imshow("Lunar Lander", frame_bgr)

        # Wait for a short period and allow window updates
        if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to quit
            break
        
        # Exit if the episode is finished or truncated
        if done or truncated:
            break
        
        time.sleep(0.01)

    # Properly close the environment
    env.close()
    cv2.destroyAllWindows()  # Close OpenCV windows

if __name__ == "__main__":
    main()
