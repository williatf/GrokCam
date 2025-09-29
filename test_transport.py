import time
from control import tcControl

def run_tests():
    """Initializes tcControl and runs a series of tests on the film transport."""
    
    print("--- Starting Film Transport Tests ---")
    
    try:
        # Initialize the control class
        control = tcControl()
        print("tcControl initialized successfully.")

        # --- Test 1: LED ---
        print("\n--- Test 1: LED ---")
        print("Turning LED ON...")
        control.light_on()
        time.sleep(2)
        print("Turning LED OFF...")
        control.light_off()
        time.sleep(1)
        print("LED test complete.")

        # --- Test 2: Stepper Motor Forward ---
        print("\n--- Test 2: Stepper Motor Forward ---")
        print("Moving stepper motor forward 100 steps.")
        control.steps_forward(100)
        time.sleep(1)
        print("Stepper motor forward test complete.")

        # --- Test 3: Stepper Motor Backward ---
        print("\n--- Test 3: Stepper Motor Backward ---")
        print("Moving stepper motor backward 100 steps.")
        control.steps_back(100)
        time.sleep(1)
        print("Stepper motor backward test complete.")

        # --- Test 4: Take-Up Pulse ---
        print("\n--- Test 4: Take-Up Pulse ---")
        print("Sending a pulse to the take-up reel.")
        control.takeup_pulse()
        time.sleep(1)
        print("Take-up pulse test complete.")

        # --- Test 5: Take-Up Pulse via steps_forward() ---
        # NOTE: This test depends on your `self.steps_taken` logic in tcControl.py
        # Your current logic has a pulse every 550 steps.
        # Let's run a full cycle to confirm it triggers.
        print("\n--- Test 5: Take-Up Pulse via steps_forward() ---")
        print("Moving stepper forward 550 steps to trigger take-up pulse.")
        # Reset steps_taken to simulate a fresh run
        control.steps_taken = 0 
        control.steps_forward(550)
        print("Expected take-up pulse should have occurred.")
        time.sleep(1)
        print("Automatic take-up test complete.")

    except Exception as e:
        print(f"\n!!! An error occurred during testing: {e}")
        
    finally:
        # --- Cleanup ---
        print("\n--- Cleaning up ---")
        control.clean_up()
        print("Cleanup complete. All systems off.")
        print("\n--- All tests finished ---")

if __name__ == "__main__":
    run_tests()
