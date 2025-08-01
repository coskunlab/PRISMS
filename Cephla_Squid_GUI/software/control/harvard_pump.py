import serial
import time
import sys

class HarvardPump:
    """
    Updated class using commands from the PHD Ultra manual (Rev 1.0).
    Uses irun/wrun instead of DIR + RUN.
    """
    def __init__(self, port, baudrate=19200, timeout=1, address=None):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.address = address
        self.ser = None
        # Prompt includes address if address is set, otherwise just ':'
        self.prompt = f"{self.address}:".encode('ascii') if self.address else b':'
        # Handle case where address is '00' - prompt might still just be ':'
        if self.address == "00":
            # Based on testing, '00:' seems common for address 00 responses
            self.prompt = b'00:'


    def connect(self):
        """Establish serial connection."""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                # Manual page 50 specifies 1 Stop Bit for RS-232, let's try that
                # Your pump might be configured differently, but manual says 1
                stopbits=serial.STOPBITS_ONE, # *** CHANGED based on manual p.50 ***
                bytesize=serial.EIGHTBITS,
                timeout=self.timeout
            )
            print(f"Connected to pump on {self.port} at {self.baudrate} baud.")
            self.ser.flushInput()
            self.ser.flushOutput()
            time.sleep(0.1)
            # Send version command without address [cite: 171] (usually doesn't need one)
            response = self.send_command("VER", include_address=False)
            if not response:
                 print("Warning: No response received for initial VER command.")
            else:
                 print(f"Pump Version Response: {response}")
                 if "3.0.9" not in response:
                     print(f"Warning: Detected firmware {response}, expected 3.0.9. Commands might differ.")
                 # Update prompt expectation based on VER response if needed (e.g., if address 00 returns ':')
                 if self.address == "00" and not response.endswith(':'):
                     # If VER for address 00 didn't end with ':', maybe it's just ':'
                     # Or if VER response included '00:', stick with that. Let's trust the test output.
                     pass # Keeping prompt as b'00:' based on previous successful DIAM/RATE tests
                 elif not self.address and not response.endswith(':'):
                     print("Warning: Unexpected prompt format after VER.")


        except serial.SerialException as e:
            print(f"Error connecting to pump on {self.port}: {e}")
            sys.exit(1)

    def disconnect(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            print("Attempting to stop pump before disconnecting...")
            self.stop() # stop() will use address if set
            time.sleep(0.5)
            self.ser.close()
            print("Disconnected from pump.")
        elif self.ser:
            print("Serial port already closed.")


    def send_command(self, command, include_address=True):
        """
        Sends a command to the pump, optionally prepending the address,
        and returns the response.
        """
        if not self.ser or not self.ser.is_open:
            print("Error: Serial port not open.")
            return None

        full_command_str = command
        # Prepend address if needed and available [cite: 171]
        if include_address and self.address is not None:
            full_command_str = f"{self.address} {command}"

        try:
            full_command = (full_command_str + '\r').encode('ascii')
            print(f"Sending: {full_command!r}")
            # Clear input buffer before sending
            self.ser.reset_input_buffer()
            self.ser.write(full_command)
            time.sleep(0.1) # Give pump time to process

            response_bytes = b""
            start_time = time.time()
            # Read until expected prompt or timeout
            while time.time() - start_time < self.timeout:
                 if self.ser.in_waiting > 0:
                      byte = self.ser.read(1)
                      response_bytes += byte
                      # Check if the response ends with the specific expected prompt
                      if response_bytes.endswith(self.prompt):
                           break
                 else:
                      time.sleep(0.05)

            # Decode response, strip prompt and whitespace
            response_str = response_bytes.replace(self.prompt, b'').decode('ascii', errors='ignore').strip()
            print(f"Received Raw: {response_bytes!r}") # Show raw response for debugging
            print(f"Received Parsed: {response_str!r}")

            # Use manual's error format [cite: 180, 182]
            if 'command error' in response_str.lower() or \
               'argument error' in response_str.lower() or \
               '?' in response_str:
                 print(f"Warning: Pump reported an error: {response_str}")
            elif not response_bytes.endswith(self.prompt) and len(response_bytes) > 0:
                 # Didn't get the expected prompt, but got something else
                 print(f"Warning: Unexpected response format or missing prompt. Got: {response_bytes!r}")
            elif len(response_bytes) == 0:
                 print(f"Warning: No response received from pump for command '{full_command_str}'.")


            return response_str # Return the cleaned string part of the response

        except serial.SerialException as e:
            print(f"Serial communication error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during send/receive: {e}")
            return None

    def set_diameter(self, diameter_mm):
        """Sets the syringe diameter using DIAM command[cite: 220]."""
        # According to manual, DIAM is Quick Start mode only [cite: 220]
        return self.send_command(f"diameter {diameter_mm:.2f}") # Use lowercase as per manual [cite: 220]

    def set_infusion_rate(self, rate_value, rate_unit_key):
        """Sets the infusion rate using irate command[cite: 224]."""
        if rate_unit_key not in RATE_UNIT_CODES:
             print(f"Error: Invalid rate unit key '{rate_unit_key}'")
             return None
        pump_unit_code = RATE_UNIT_CODES[rate_unit_key]
        # According to manual, irate is Quick Start mode only [cite: 224]
        return self.send_command(f"irate {rate_value:.3f} {pump_unit_code}") # Use lowercase [cite: 225]

    # --- set_direction method removed as it doesn't exist ---

    def start_infuse(self):
        """Starts the pump infusing using irun command[cite: 214]."""
        # According to manual, irun is Quick Start mode only [cite: 214]
        return self.send_command("irun") # Use lowercase [cite: 214]

    def start_withdraw(self):
        """Starts the pump withdrawing using wrun command[cite: 217]."""
         # According to manual, wrun is Quick Start mode only [cite: 217]
        return self.send_command("wrun") # Use lowercase [cite: 217]

    def stop(self):
        """Stops the pump using stop or stp command[cite: 215]."""
        return self.send_command("stop") # Use lowercase [cite: 215]

    def get_status(self):
        """Queries the pump status (basic example using rate query)[cite: 228]."""
        # crate (current rate) might be more useful if running [cite: 218]
        # irate queries the *set* infusion rate [cite: 224]
        return self.send_command("irate")


# --- Main Execution ---
print("--- Harvard Pump PhD Ultra Infusion Script (Using Manual Commands) ---")
print("*** The pump will run indefinitely until manually stopped (Ctrl+C) ***")

# --- Configuration ---
# !! MODIFY THESE VALUES FOR YOUR SETUP !!
SERIAL_PORT = '/dev/ttyACM0' # Matched from your output
BAUD_RATE = 19200
PUMP_ADDRESS = "00"          # Confirmed address prefix
SYRINGE_DIAMETER_MM = 29.20  # Matched from your output
INFUSION_RATE_VALUE = 10    # Simplified rate value for testing
# *** UPDATED: Match manual examples (m/m, u/m, m/h, u/h) ***
# Using 'm/m' for mL/min as per manual note [cite: 232]
INFUSION_RATE_UNIT = 'UM'
# --- End Configuration ---

# Map user-friendly units to pump command units based on manual [cite: 232]
RATE_UNIT_CODES = {
    'ul/s': 'u/s', 'ml/s': 'm/s',
    'ul/m': 'u/m', 'ml/m': 'm/m',
    'ul/h': 'u/h', 'ml/h': 'm/h',
    # Add synonyms if desired
    'UM': 'u/m', 'MM': 'm/m', 'UH': 'u/h', 'MH': 'm/h'
}

# Validate configuration unit key
if INFUSION_RATE_UNIT not in RATE_UNIT_CODES:
    # Try matching case-insensitively if direct match failed
    found_match = False
    for key in RATE_UNIT_CODES:
        if key.lower() == INFUSION_RATE_UNIT.lower():
            INFUSION_RATE_UNIT = key # Use the correct case from the dictionary key
            found_match = True
            break
    if not found_match:
        print(f"FATAL: Invalid INFUSION_RATE_UNIT: '{INFUSION_RATE_UNIT}'. Valid keys are: {list(RATE_UNIT_CODES.keys())}")
        sys.exit(1)

pump = HarvardPump(SERIAL_PORT, BAUD_RATE, address=PUMP_ADDRESS)

try:
    pump.connect()

    print("\n--- Configuring Pump ---")
    # Use methods with lowercase commands as per manual examples
    pump.set_diameter(SYRINGE_DIAMETER_MM)
    time.sleep(0.2)
    pump.set_infusion_rate(INFUSION_RATE_VALUE, INFUSION_RATE_UNIT)
    time.sleep(0.2)
    # No direction command needed

    print("\n--- Starting Infusion (will run indefinitely) ---")
    # input("Press Enter to start the infusion...")

    pump.start_infuse() # Use the specific infusion start command
    print("Pump infuse command sent.")
    print("Infusing continuously. Press Ctrl+C to stop.")

    while True:
        # Optionally query status periodically here if needed
        pump.get_status()
        time.sleep(5)
        # time.sleep(1) # Keep script alive

except Exception as e:
    print(f"\nAn unexpected error occurred in the main block: {e}")
finally:
    print("\n--- Script Ending: Ensuring pump is stopped and disconnected ---")
    pump.disconnect() # disconnect() calls stop()
    print("\n--- Script Finished ---")