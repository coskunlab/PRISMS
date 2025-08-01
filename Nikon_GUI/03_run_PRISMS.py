import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import os
from datetime import date, datetime
from sklearn.linear_model import LinearRegression
import shutil
import time
import smtplib
from email.message import EmailMessage

class NikonMacroGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nikon Macro Generator")
        self.geometry("600x900")

        # Style
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TLabel', font=('Helvetica', 10))
        style.configure('TButton', font=('Helvetica', 10, 'bold'))
        style.configure('TEntry', font=('Helvetica', 10))
        style.configure('TFrame', padding=10)
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))

        # Main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Initialize crash detection variables
        self.acquisition_start_time = None
        self.last_activity_time = None

        # --- User Inputs ---
        self.create_widgets(main_frame)
        self.on_workflow_change() # Set initial UI state
        self.on_email_toggle() # Set initial email UI state

    def create_widgets(self, parent):
        # --- Workflow and File Setup ---
        setup_frame = ttk.LabelFrame(parent, text="Workflow & File Setup", padding=10)
        setup_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)

        ttk.Label(setup_frame, text="Workflow Mode:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.workflow_mode_var = tk.StringVar(value="Tile Scan (Tissue)")
        self.workflow_combobox = ttk.Combobox(
            setup_frame,
            textvariable=self.workflow_mode_var,
            values=["Tile Scan (Tissue)", "Multipoint (Well Plate)"],
            state="readonly"
        )
        self.workflow_combobox.grid(row=0, column=1, columnspan=2, sticky="ew", padx=5)
        self.workflow_combobox.bind("<<ComboboxSelected>>", self.on_workflow_change)

        ttk.Label(setup_frame, text="Experiment XML File:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.xml_file_var = tk.StringVar()
        ttk.Entry(setup_frame, textvariable=self.xml_file_var, width=50, state="readonly").grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(setup_frame, text="Select XML File...", command=self.select_and_setup_paths).grid(row=1, column=2, padx=5)

        ttk.Label(setup_frame, text="Determined Base Path:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.base_path_var = tk.StringVar()
        ttk.Entry(setup_frame, textvariable=self.base_path_var, width=50, state="readonly").grid(row=2, column=1, sticky="ew", padx=5)
        
        setup_frame.columnconfigure(1, weight=1)

        # --- Z-Stack Settings ---
        z_stack_frame = ttk.LabelFrame(parent, text="Z-Stack Settings", padding=10)
        z_stack_frame.grid(row=1, column=0, sticky="nsew", pady=5, padx=5)

        self.z_step_num_var = tk.StringVar(value="40")
        ttk.Label(z_stack_frame, text="Number of Slices:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(z_stack_frame, textvariable=self.z_step_num_var).grid(row=0, column=1, pady=2)

        self.z_step_size_var = tk.StringVar(value="0.5")
        ttk.Label(z_stack_frame, text="Step Size (µm):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(z_stack_frame, textvariable=self.z_step_size_var).grid(row=1, column=1, pady=2)

        # --- Autofocus Settings ---
        autofocus_frame = ttk.LabelFrame(parent, text="Autofocus Settings", padding=10)
        autofocus_frame.grid(row=1, column=1, sticky="nsew", pady=5, padx=5)

        self.focus_count_var = tk.StringVar(value="100")
        ttk.Label(autofocus_frame, text="Number of Z-planes:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(autofocus_frame, textvariable=self.focus_count_var).grid(row=0, column=1, pady=2)

        self.focus_step_var = tk.StringVar(value="2")
        ttk.Label(autofocus_frame, text="Step Size (µm):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(autofocus_frame, textvariable=self.focus_step_var).grid(row=1, column=1, pady=2)
        
        self.channel_list = ['DAPI', 'FITC', 'TRITC', 'Cy5']
        self.focus_chan_var = tk.StringVar(value="DAPI")
        ttk.Label(autofocus_frame, text="Focus Channel:").grid(row=2, column=0, sticky="w", pady=2)
        self.focus_chan_combo = ttk.Combobox(autofocus_frame, textvariable=self.focus_chan_var, values=self.channel_list, state="readonly")
        self.focus_chan_combo.grid(row=2, column=1, pady=2)
        
        self.wait_time_var = tk.StringVar(value="0.5")
        ttk.Label(autofocus_frame, text="Wait Time (min):").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(autofocus_frame, textvariable=self.wait_time_var).grid(row=3, column=1, pady=2)

        # --- Acquisition Settings ---
        acq_frame = ttk.LabelFrame(parent, text="Acquisition Settings", padding=10)
        acq_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)

        ttk.Label(acq_frame, text="Overlap Fraction:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.overlap_frac_var = tk.StringVar(value="0.15")
        self.overlap_entry = ttk.Entry(acq_frame, textvariable=self.overlap_frac_var)
        self.overlap_entry.grid(row=0, column=1, padx=5, pady=2)

        self.stitch_mode_var = tk.StringVar(value="MIP")
        ttk.Label(acq_frame, text="Stitching Mode:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        self.stitch_mode_combo = ttk.Combobox(acq_frame, textvariable=self.stitch_mode_var, values=["MIP", "BestFocus"], state="readonly")
        self.stitch_mode_combo.grid(row=0, column=3, padx=5, pady=2)

        self.objective_var = tk.StringVar(value="40X (Oil)")
        ttk.Label(acq_frame, text="Objective:").grid(row=0, column=4, sticky="w", padx=5, pady=2)
        self.objective_combo = ttk.Combobox(acq_frame, textvariable=self.objective_var, values=['20X', '40X (Air)', '40X (Oil)', '60X'], state="readonly")
        self.objective_combo.grid(row=0, column=5, padx=5, pady=2)
        
        # --- Multipoint Grid Settings (Initially hidden) ---
        self.multipoint_grid_frame = ttk.LabelFrame(parent, text="Multipoint Grid Settings", padding=10)
        self.multipoint_grid_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)

        self.tiles_x_var = tk.StringVar(value="3")
        ttk.Label(self.multipoint_grid_frame, text="Grid Tiles in X:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(self.multipoint_grid_frame, textvariable=self.tiles_x_var).grid(row=0, column=1, padx=5, pady=2)

        self.tiles_y_var = tk.StringVar(value="3")
        ttk.Label(self.multipoint_grid_frame, text="Grid Tiles in Y:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        ttk.Entry(self.multipoint_grid_frame, textvariable=self.tiles_y_var).grid(row=0, column=3, padx=5, pady=2)

        # --- Channel Exposures ---
        exposure_frame = ttk.LabelFrame(parent, text="Channel Exposures (ms)", padding=10)
        exposure_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
        
        self.channel_vars = {}
        default_exposures = ["25", "100", "1000", "1000"]
        for i, (channel, exp) in enumerate(zip(self.channel_list, default_exposures)):
            ttk.Label(exposure_frame, text=f"{channel}:").grid(row=0, column=i*2, sticky="w", padx=5)
            var = tk.StringVar(value=exp)
            ttk.Entry(exposure_frame, textvariable=var, width=8).grid(row=0, column=i*2+1, padx=5)
            self.channel_vars[channel] = var

        # --- Generate Button ---
        generate_button = ttk.Button(parent, text="Generate Macro Command", command=self.generate_macro)
        generate_button.grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")

        # --- Email Alert Settings ---
        email_frame = ttk.LabelFrame(parent, text="Email Alert Settings", padding=10)
        email_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=5)

        self.email_enabled_var = tk.BooleanVar(value=False)
        self.email_checkbox = ttk.Checkbutton(
            email_frame, 
            text="Send email alerts on crash detection", 
            variable=self.email_enabled_var,
            command=self.on_email_toggle
        )
        self.email_checkbox.grid(row=0, column=0, columnspan=4, sticky="w", pady=2)

        # Email input fields (initially hidden)
        ttk.Label(email_frame, text="Email Address:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.email_address_var = tk.StringVar(value="Outlook Email Address")
        self.email_entry = ttk.Entry(email_frame, textvariable=self.email_address_var, width=30)
        self.email_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)

        ttk.Label(email_frame, text="Email Password:").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        self.email_password_var = tk.StringVar(value="")
        self.email_password_entry = ttk.Entry(email_frame, textvariable=self.email_password_var, show="*", width=20)
        self.email_password_entry.grid(row=1, column=3, sticky="ew", padx=5, pady=2)

        email_frame.columnconfigure(1, weight=1)
        email_frame.columnconfigure(3, weight=1)

        # --- Output ---
        output_frame = ttk.LabelFrame(parent, text="Output Command / Status", padding=10)
        output_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=5)

        self.output_text = tk.Text(output_frame, height=4, wrap=tk.WORD, font=('Courier', 10))
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

    def on_workflow_change(self, event=None):
        """Shows or hides the multipoint grid settings based on workflow mode."""
        mode = self.workflow_mode_var.get()
        if "Multipoint" in mode:
            self.multipoint_grid_frame.grid()
            self.overlap_entry.config(state="normal")
        else: # Tile Scan
            self.multipoint_grid_frame.grid_remove()
            self.overlap_entry.config(state="normal")

    def on_email_toggle(self):
        """Shows or hides email input fields based on checkbox state."""
        if self.email_enabled_var.get():
            # Show email fields
            self.email_entry.grid()
            self.email_password_entry.grid()
        else:
            # Hide email fields
            self.email_entry.grid_remove()
            self.email_password_entry.grid_remove()

    def select_and_setup_paths(self, event=None):
        filepath = filedialog.askopenfilename(
            title="Select Experiment XML File",
            filetypes=(("XML files", "*.xml"), ("All files", "*.*"))
        )
        if not filepath: return
        xml_path = Path(filepath)
        parent_dir = xml_path.parent
        if parent_dir.name.lower() == 'macros':
            base_path = parent_dir.parent
            final_xml_path = xml_path
        else:
            base_path = parent_dir
            macro_path = base_path / 'Macros'
            macro_path.mkdir(exist_ok=True)
            target_path = macro_path / xml_path.name
            if not target_path.exists():
                try:
                    shutil.move(xml_path, target_path)
                    messagebox.showinfo("File Moved", f"XML file moved to:\n{target_path}")
                except Exception as e:
                    messagebox.showerror("File Move Error", f"Could not move XML file.\n{e}")
                    return
            final_xml_path = target_path
        self.base_path_var.set(str(base_path))
        self.xml_file_var.set(str(final_xml_path))

    def collect_common_inputs(self):
        if not self.base_path_var.get() or not self.xml_file_var.get():
            messagebox.showerror("Error", "Please select an experiment XML file first.")
            return None
        inputs = {}
        base_path = Path(self.base_path_var.get())
        inputs['xml_file'] = Path(self.xml_file_var.get())
        if not base_path.exists() or not inputs['xml_file'].exists():
            raise ValueError("Base Path or XML File does not exist.")
        
        # --- Local Path Logic ---
        e_drive = Path('E:/')
        run_path = base_path # Default run_path to base_path
        
        if e_drive.exists():
            local_path = e_drive / ' '.join(base_path.parts[-2:])
            run_path = local_path # Set run_path to local if E: exists
            
            # Only copy if source and destination are different
            if base_path.resolve() != run_path.resolve():
                run_path.mkdir(parents=True, exist_ok=True)
                print(f"E: drive found. Using local path for acquisition: {run_path}")
                try:
                    # Copy initial files if needed, be careful not to overwrite existing data
                    shutil.copytree(base_path, run_path, dirs_exist_ok=True)
                    print(f"Copied experiment files from {base_path} to {run_path}")
                except Exception as e:
                    print(f"Could not copy from base to local path: {e}")
            else:
                print("Local path is the same as base path. Skipping file copy.")
        else:
            print(f"E: drive not found. Using base path for acquisition: {run_path}")

        inputs['base_path'] = base_path
        inputs['run_path'] = run_path
        
        inputs['z_step_num'] = int(self.z_step_num_var.get())
        inputs['z_step_size'] = float(self.z_step_size_var.get())
        inputs['focus_count'] = int(self.focus_count_var.get())
        inputs['focus_step'] = float(self.focus_step_var.get())
        inputs['focus_chan'] = self.focus_chan_var.get()
        inputs['wait_time'] = float(self.wait_time_var.get())
        inputs['objective_str'] = self.objective_var.get()
        inputs['overlap_frac'] = float(self.overlap_frac_var.get())
        
        if "Multipoint" in self.workflow_mode_var.get():
            inputs['tiles_x'] = int(self.tiles_x_var.get())
            inputs['tiles_y'] = int(self.tiles_y_var.get())

        channels_df = pd.DataFrame()
        channels_df['Filter'] = self.channel_vars.keys()
        exposures = [int(self.channel_vars[ch].get()) for ch in channels_df['Filter']]
        channels_df['Exposure'] = exposures
        inputs['channels_df'] = channels_df.loc[channels_df['Exposure'] > 0].reset_index(drop=True)
        if inputs['focus_chan'] not in inputs['channels_df']['Filter'].values:
            raise ValueError(f"Focus channel '{inputs['focus_chan']}' not in active channels.")
        
        all_res = {'20X': 0.325, '40X (Air)': 0.1663, '40X (Oil)': 0.1625, '60X': 0.1083}
        if inputs['objective_str'] not in all_res: 
            raise ValueError("Invalid objective selected.")
        inputs['img_res'] = all_res[inputs['objective_str']]

        inputs['macro_save_path'] = run_path / 'Macros'
        inputs['img_save_path'] = run_path / '00 RAW'
        inputs['macro_save_path'].mkdir(exist_ok=True)
        inputs['img_save_path'].mkdir(exist_ok=True)
        for f in inputs['macro_save_path'].glob('*.mac'): os.remove(f)
        return inputs

    def generate_macro(self):
        try:
            inputs = self.collect_common_inputs()
            if not inputs: return
            
            # Print all user selections to terminal for confirmation
            print("\n" + "="*60)
            print("USER SELECTIONS SUMMARY")
            print("="*60)
            print(f"Workflow Mode: {self.workflow_mode_var.get()}")
            print(f"XML File: {inputs['xml_file']}")
            print(f"Base Path: {inputs['base_path']}")
            print(f"Run Path: {inputs['run_path']}")
            print(f"Objective: {inputs['objective_str']}")
            print(f"Image Resolution: {inputs['img_res']} µm/pixel")
            print(f"Overlap Fraction: {inputs['overlap_frac']}")
            print(f"Stitching Mode: {self.stitch_mode_var.get()}")
            
            print("\nZ-Stack Settings:")
            print(f"  Number of Slices: {inputs['z_step_num']}")
            print(f"  Step Size: {inputs['z_step_size']} µm")
            
            print("\nAutofocus Settings:")
            print(f"  Number of Z-planes: {inputs['focus_count']}")
            print(f"  Step Size: {inputs['focus_step']} µm")
            print(f"  Focus Channel: {inputs['focus_chan']}")
            print(f"  Wait Time: {inputs['wait_time']} min")
            
            if "Multipoint" in self.workflow_mode_var.get():
                print("\nMultipoint Grid Settings:")
                print(f"  Grid Tiles X: {inputs['tiles_x']}")
                print(f"  Grid Tiles Y: {inputs['tiles_y']}")
            
            print("\nChannel Exposures:")
            for _, row in inputs['channels_df'].iterrows():
                print(f"  {row['Filter']}: {row['Exposure']} ms")
            
            print(f"\nMacro Save Path: {inputs['macro_save_path']}")
            print(f"Image Save Path: {inputs['img_save_path']}")
            print("="*60)
            
            tree = ET.parse(inputs['xml_file'])
            root = tree.getroot()
            x_coords = [eval(elem.attrib['value']) for elem in root.findall('.//dPosX')[0]]
            y_coords = [eval(elem.attrib['value']) for elem in root.findall('.//dPosY')[0]]
            z_coords = [eval(elem.attrib['value']) for elem in root.findall('.//dPosZ')[0]]
            pts = np.stack([x_coords, y_coords, z_coords], axis=1)
            print(f"Number of points loaded from XML: {len(pts)}")
            print("="*60 + "\n")
            
            if "Tile Scan" in self.workflow_mode_var.get():
                self.generate_tile_scan_macros(inputs, pts)
            else:
                self.generate_multipoint_macros(inputs, pts)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{e}")
            import traceback
            traceback.print_exc()

    def display_output_message(self, message):
        """Display a message in the GUI output text box."""
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, message)
        self.update()  # Force GUI update
    
    def append_output_message(self, message):
        """Append a message to the GUI output text box."""
        self.output_text.insert(tk.END, "\n" + message)
        self.output_text.see(tk.END)  # Scroll to bottom
        self.update()  # Force GUI update

    def detectNikonCrash(self, timeWaitStart, img_save_path, macro_save_path):
        """
        Detect if NIS Elements software crashed based on timing.
        Returns True if crash detected, False otherwise.
        """
        timeNow = time.time()
        if timeNow - timeWaitStart > 10 * 60:  # 10 mins
            # Software likely crashed. Delete previous incomplete channel stack and print run line
            crash_message = f'Detected NIS Elements crash at {str(datetime.now())}'
            print(crash_message)
            self.append_output_message(crash_message)
            
            try:
                # Find recent ND2 files
                img_files = [f for f in img_save_path.glob('*.nd2')]
                if img_files:
                    # Get last date modified
                    recent_file = max(img_files, key=os.path.getctime)
                    parts = recent_file.stem.split('_')
                    
                    if len(parts) >= 3:
                        x_curr = parts[1] if parts[1].startswith('x') else parts[2]
                        y_curr = parts[2] if parts[2].startswith('y') else parts[1]
                        
                        if '_zScan' in recent_file.stem:  # autofocus file
                            # Find autofocus macro file for this tile
                            file_name = [f for f in macro_save_path.glob('*.mac') if 
                                        f'_{x_curr}' in f.name and 
                                        f'_{y_curr}' in f.name and 
                                        'zScan' in f.name]
                            if file_name:
                                # Remove relative motion from autofocus macro bc it is already at position
                                with open(file_name[0], 'r') as file:
                                    content = file.readlines()
                                for kk, line in enumerate(content):
                                    if 'StgMoveXY' in line:
                                        content[kk] = '// ' + line  # comment out XY movement
                                # Write macro file
                                with open(file_name[0], 'w') as file:
                                    file.writelines(content)
                        else:  # normal channel stack file
                            # Find macro file for this tile
                            file_name = [f for f in macro_save_path.glob('*.mac') if 
                                        f'_{x_curr}' in f.name and 
                                        f'_{y_curr}' in f.name and 
                                        'zScan' not in f.name]
                        
                        if file_name:
                            # Print run line for user to copy/paste
                            run_command = f'RunMacro("{str(file_name[0])}");'
                            print()
                            print(run_command)
                            print()
                            
                            # Display in GUI
                            crash_output = f"CRASH DETECTED - Copy this command:\n{run_command}"
                            self.display_output_message(crash_output)
                            
                            # Send email alert if enabled
                            if self.email_enabled_var.get():
                                self.send_crash_email(recent_file.name)
                            else:
                                no_email_message = "Email alerts disabled - no email sent"
                                print(no_email_message)
                                self.append_output_message(no_email_message)
                            
                            # Delete incomplete file
                            delete_message = f'Deleting incomplete image stack...{recent_file.name}'
                            print(delete_message)
                            self.append_output_message(delete_message)
                            try:
                                os.remove(recent_file)
                                success_message = f'Successfully deleted {recent_file.stem}'
                                print(success_message)
                                self.append_output_message(success_message)
                            except Exception as e:
                                error_message = f'Failed to delete {recent_file.name}: {e}'
                                print(error_message)
                                self.append_output_message(error_message)
                        
            except Exception as e:
                error_message = f"Error in crash detection: {e}"
                print(error_message)
                self.append_output_message(error_message)
            
            return True
        return False
    
    def send_crash_email(self, filename):
        """Send email alert for software crash using user-provided credentials."""
        try:
            sender = self.email_address_var.get()
            password = self.email_password_var.get()
            recipient = sender  # Send to same email address
            subject = "Detected Nikon NIS Elements software crash"
            
            if not sender or not password:
                error_message = "Email address and password must be provided to send alerts"
                print(error_message)
                self.append_output_message(error_message)
                return
            
            email = EmailMessage()
            email["From"] = sender
            email["To"] = recipient
            email["Subject"] = subject
            
            # Write body of email
            body = f'Nikon NIS Elements software crashed at {str(datetime.now())}. ' \
                   f'Incomplete file: {filename}. Please copy/paste run macro line to resume imaging.\n'
            email.set_content(body)
            
            # Send the email
            smtp_server = "smtp-mail.outlook.com"
            port = 587
            email_message = f'Sending email alert to {recipient} at {str(datetime.now())}'
            print(email_message)
            self.append_output_message(email_message)
            
            with smtplib.SMTP(smtp_server, port) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(email)
                
            success_message = "Email alert sent successfully"
            print(success_message)
            self.append_output_message(success_message)
                
        except Exception as e:
            error_message = f"Error sending email: {e}"
            print(error_message)
            self.append_output_message(error_message)

    def finalize_and_chain_macros(self, inputs):
        macro_save_path = inputs['macro_save_path']
        all_dapi_macros = sorted(macro_save_path.glob(f"*_{inputs['focus_chan']}_zScan.mac"), key=lambda p: int(p.name.split('_')[0]))
        all_acq_macros = sorted([p for p in macro_save_path.glob("*.mac") if "_zScan" not in p.name], key=lambda p: int(p.name.split('_')[0]))
        if not all_dapi_macros: raise ValueError("No macro files were generated.")
        for i in range(len(all_acq_macros) - 1):
            with open(all_acq_macros[i], "a") as f:
                f.write(f'RunMacro("{all_dapi_macros[i+1]}");\n')
        with open(all_acq_macros[-1], "a") as f:
            f.write('Stg_Light_SetIrisIntensity("EPI", 0);\n')
            f.write('StgMoveMainZ(-4000, 1); // final\n')
        first_macro_path = all_dapi_macros[0]
        output_command = f'RunMacro("{first_macro_path}");'
        self.display_output_message(output_command)
        
        # Start acquisition timing for crash detection
        self.start_acquisition_timing()
        
        messagebox.showinfo("Success", "Macro files generated successfully!")

    def write_macro_pair(self, f_dapi, f_acq, params):
        if params['is_first']:
            f_dapi.write(f"// Date: {date.today().strftime('%b-%d-%Y')}\n")
            f_dapi.write(f"// Z Stack = {params['z_step_num']} slices, Z step size = {params['z_step_size']} um\n\n")
            f_dapi.write('ShowImageInfoBeforeSaveAs(0);\nLUTs_KeepAutoScale(1);\n')
        for _, row in params['channels_df'].iterrows():
            f_dapi.write(f'SelectOptConf("{row["Filter"]}");\nCameraSet_Exposure(1, {row["Exposure"]});\n')
        f_dapi.write('Stg_SetShutterStateEx("EPI", 0);\n')
        f_dapi.write(f'ND_LoadExperiment("{params["xml_file"]}");\n\n')
        f_dapi.write(params['move_cmd'])
        f_dapi.write(f"StgMoveMainZ({params['z']}, 0);\n")
        f_dapi.write('Stg_SetShutterStateEx("EPI", 1);\n')
        f_dapi.write(f'SelectOptConf("{params["focus_chan"]}");\nLiveSync();\n')
        nd2_dapi = f"{params['img_save_path']}\\{params['file_prefix']}_{params['focus_chan']}_zScan.nd2"
        f_dapi.write(f'ND_DefineExperiment(0, 0, 1, 0, 1, "{nd2_dapi}","",0,0,0,0);\n')
        f_dapi.write(f'ND_SetZSeriesExp(2, 0, {params["z"]}, 0, {params["focus_step"]}, {params["focus_count"]}, 0, 0, "", "", "");\n')
        f_dapi.write('ND_RunExperiment(0);\nCloseCurrentDocument(0);\n\n')
        f_dapi.write('Stg_SetShutterStateEx("EPI", 0);\n')
        f_dapi.write(f'WaitText({params["wait_time"] * 60}, "Waiting for Python to compute focus plane...");\n')
        f_dapi.write(f'RunMacro("{params["acq_macro_path"]}");\n')
        for _, row in params['channels_df'].iterrows():
            f_acq.write(f'SelectOptConf("{row["Filter"]}");\nCameraSet_Exposure(1, {row["Exposure"]});\n')
        f_acq.write('Stg_SetShutterStateEx("EPI", 0);\n')
        f_acq.write(f"StgMoveMainZ({params['z']}, 0);\n")
        f_acq.write('Stg_SetShutterStateEx("EPI", 1);\n')
        nd2_acq = f"{params['img_save_path']}\\{params['file_prefix']}_chStack.nd2"
        f_acq.write(f'ND_DefineExperiment(0, 0, 1, 1, 1, "{nd2_acq}","",0,0,0,0);\n')
        f_acq.write(f'ND_SetZSeriesExp(2, 0, {params["z"]}, 0, {params["z_step_size"]}, {params["z_step_num"]}, 0, 0, "", "", "");\n')
        f_acq.write('ND_RunExperiment(0);\nCloseCurrentDocument(0);\n\n')

    def generate_tile_scan_macros(self, inputs, pts):
        dimY, dimX = 2304, 2304
        minY, maxY = np.min(pts[:, 1]), np.max(pts[:, 1])
        minX, maxX = np.min(pts[:, 0]), np.max(pts[:, 0])
        step_dist_Y = dimY * inputs['img_res'] * (1 - inputs['overlap_frac'])
        step_dist_X = dimX * inputs['img_res'] * (1 - inputs['overlap_frac'])
        y_range = np.arange(minY, maxY + dimY * inputs['img_res'], step_dist_Y)
        x_range = np.arange(minX, maxX + dimX * inputs['img_res'], step_dist_X)
        model = LinearRegression().fit(pts[:, [0, 1]], pts[:, 2])
        a, b = model.coef_
        c = model.intercept_
        macro_file_count = 0
        for ii, yCurr in enumerate(y_range):
            xRangeSub = np.flip(x_range) if (ii % 2) != 0 else x_range
            xScanDir = -1 if (ii % 2) != 0 else 1
            for jj, xCurr in enumerate(xRangeSub):
                macro_file_count += 1
                file_prefix = f"Tile_x{jj + 1:03d}_y{ii + 1:03d}"
                dapi_macro_path = inputs['macro_save_path'] / f"{macro_file_count:03d}_{file_prefix}_{inputs['focus_chan']}_zScan.mac"
                macro_file_count += 1
                acq_macro_path = inputs['macro_save_path'] / f"{macro_file_count:03d}_{file_prefix}.mac"
                params = {**inputs, 'z': a * xCurr + b * yCurr + c, 'is_first': (ii == 0 and jj == 0),
                          'file_prefix': file_prefix, 'acq_macro_path': acq_macro_path}
                if ii == 0 and jj == 0:
                    params['move_cmd'] = f"StgMoveXY({xCurr}, {yCurr}, 0);\n"
                else:
                    yJump = step_dist_Y if jj == 0 else 0
                    xJump = 0 if jj == 0 else step_dist_X * xScanDir
                    params['move_cmd'] = f"StgMoveXY({xJump}, {yJump}, 1);\n"
                with open(dapi_macro_path, "w") as f_dapi, open(acq_macro_path, "w") as f_acq:
                    self.write_macro_pair(f_dapi, f_acq, params)
        self.finalize_and_chain_macros(inputs)

    def generate_multipoint_macros(self, inputs, pts):
        dimY, dimX = 2304, 2304
        step_dist_Y = dimY * inputs['img_res'] * (1 - inputs['overlap_frac'])
        step_dist_X = dimX * inputs['img_res'] * (1 - inputs['overlap_frac'])
        macro_file_count = 0
        for i, (px, py, pz) in enumerate(pts):
            start_x = px - ((inputs['tiles_x'] - 1) / 2) * step_dist_X
            start_y = py - ((inputs['tiles_y'] - 1) / 2) * step_dist_Y
            x_grid = np.linspace(start_x, start_x + (inputs['tiles_x'] - 1) * step_dist_X, inputs['tiles_x'])
            y_grid = np.linspace(start_y, start_y + (inputs['tiles_y'] - 1) * step_dist_Y, inputs['tiles_y'])
            for ii, yCurr in enumerate(y_grid):
                for jj, xCurr in enumerate(x_grid):
                    macro_file_count += 1
                    file_prefix = f"Point{i+1:03d}_Tile_x{jj+1:03d}_y{ii+1:03d}"
                    dapi_macro_path = inputs['macro_save_path'] / f"{macro_file_count:03d}_{file_prefix}_{inputs['focus_chan']}_zScan.mac"
                    macro_file_count += 1
                    acq_macro_path = inputs['macro_save_path'] / f"{macro_file_count:03d}_{file_prefix}.mac"
                    params = {**inputs, 'z': pz, 'is_first': (macro_file_count == 2),
                              'file_prefix': file_prefix, 'acq_macro_path': acq_macro_path,
                              'move_cmd': f"StgMoveXY({xCurr}, {yCurr}, 0);\n"}
                    with open(dapi_macro_path, "w") as f_dapi, open(acq_macro_path, "w") as f_acq:
                        self.write_macro_pair(f_dapi, f_acq, params)
        self.finalize_and_chain_macros(inputs)

    def start_acquisition_timing(self):
        """Start timing for crash detection when acquisition begins."""
        self.acquisition_start_time = time.time()
        self.last_activity_time = time.time()
        timing_message = f"Acquisition timing started at {datetime.now().strftime('%H:%M:%S')}"
        print(timing_message)
        self.append_output_message(timing_message)

if __name__ == "__main__":
    app = NikonMacroGUI()
    app.mainloop()
