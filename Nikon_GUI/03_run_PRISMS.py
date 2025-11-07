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
import asyncio
import threading
import nd2reader
import nd2 # <-- ADDED: Import the nd2 library for better dask integration
import cv2
import scipy.signal
import dask
import dask.array as da

class NikonMacroGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nikon Macro Generator & Monitor")
        self.geometry("600x950")

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

        self.create_widgets(main_frame)
        self.on_workflow_change()
        self.on_email_toggle()
        
        self.stop_monitoring_event = threading.Event()

    def create_widgets(self, parent):
        # ... (UI widget creation code remains the same) ...
        # --- Workflow and File Setup ---
        setup_frame = ttk.LabelFrame(parent, text="Workflow & File Setup", padding=10)
        setup_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(setup_frame, text="Workflow Mode:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.workflow_mode_var = tk.StringVar(value="Tile Scan (Tissue)")
        self.workflow_combobox = ttk.Combobox(setup_frame, textvariable=self.workflow_mode_var, values=["Tile Scan (Tissue)", "Multipoint (Well Plate)"], state="readonly")
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

        # --- Multipoint Grid Settings ---
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

        # --- Action Buttons ---
        action_frame = ttk.Frame(parent)
        action_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")
        self.generate_button = ttk.Button(action_frame, text="Generate Macros & Start Monitoring", command=self.start_monitoring_thread)
        self.generate_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.stop_button = ttk.Button(action_frame, text="Stop Monitoring", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        # --- Email Alert Settings ---
        email_frame = ttk.LabelFrame(parent, text="Email Alert Settings", padding=10)
        email_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=5)
        self.email_enabled_var = tk.BooleanVar(value=False)
        self.email_checkbox = ttk.Checkbutton(email_frame, text="Send email alerts on crash detection", variable=self.email_enabled_var, command=self.on_email_toggle)
        self.email_checkbox.grid(row=0, column=0, columnspan=4, sticky="w", pady=2)
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
        self.output_text = tk.Text(output_frame, height=8, wrap=tk.WORD, font=('Courier', 10), bg="black", fg="lightgreen")
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

    def on_workflow_change(self, event=None):
        mode = self.workflow_mode_var.get()
        if "Multipoint" in mode: self.multipoint_grid_frame.grid()
        else: self.multipoint_grid_frame.grid_remove()

    def on_email_toggle(self):
        if self.email_enabled_var.get():
            self.email_entry.grid(); self.email_password_entry.grid()
        else:
            self.email_entry.grid_remove(); self.email_password_entry.grid_remove()

    def select_and_setup_paths(self, event=None):
        filepath = filedialog.askopenfilename(title="Select Experiment XML File", filetypes=(("XML files", "*.xml"), ("All files", "*.*")))
        if not filepath: return
        xml_path = Path(filepath)
        parent_dir = xml_path.parent
        if parent_dir.name.lower() == 'macros': base_path = parent_dir.parent; final_xml_path = xml_path
        else:
            base_path = parent_dir
            macro_path = base_path / 'Macros'; macro_path.mkdir(exist_ok=True)
            target_path = macro_path / xml_path.name
            if not target_path.exists():
                try: shutil.move(xml_path, target_path); messagebox.showinfo("File Moved", f"XML file moved to:\n{target_path}")
                except Exception as e: messagebox.showerror("File Move Error", f"Could not move XML file.\n{e}"); return
            final_xml_path = target_path
        self.base_path_var.set(str(base_path)); self.xml_file_var.set(str(final_xml_path))

    def display_output_message(self, message):
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, message)
        self.output_text.see(tk.END)

    def append_output_message(self, message):
        self.output_text.insert(tk.END, "\n" + message)
        self.output_text.see(tk.END)
        print(message)

    def start_monitoring_thread(self):
        self.generate_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.stop_monitoring_event.clear()
        self.monitor_thread = threading.Thread(target=self.run_async_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.append_output_message("\n>>> STOPPING MONITORING... <<<")
        self.stop_monitoring_event.set()
        self.generate_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def run_async_loop(self):
        try:
            asyncio.run(self.generate_and_monitor())
        except Exception as e:
            self.append_output_message(f"\nERROR in monitoring thread: {e}")
        finally:
            self.after(100, self.stop_monitoring)

    def computeLapVar(self, plane):
        var = cv2.Laplacian(plane, cv2.CV_64F, ksize=31)
        return np.var(var)

    def findFocusLapVar(self, subStack):
        lazy_lap_var = dask.delayed(self.computeLapVar)
        lapVar = [da.from_delayed(lazy_lap_var(plane), shape=(), dtype=float) for plane in subStack]
        lapVar = da.compute(*lapVar)
        
        grad = np.gradient(lapVar)**2
        mean = np.mean(grad)
        peaks, _ = scipy.signal.find_peaks(x=grad, height=mean, distance=len(lapVar) // 3)
        
        if len(peaks) == 0:
            idxFocus = len(lapVar) // 2
        else:
            # Find peak closest to the center of the z-stack
            center_idx = len(lapVar) // 2
            closest_peak_idx = np.argmin(np.abs(peaks - center_idx))
            idxFocus = peaks[closest_peak_idx]
            
        if idxFocus >= len(lapVar) - 2:
            idxFocus = len(lapVar) - 3
        if idxFocus < 0:
            idxFocus = 0
            
        return idxFocus

    async def generate_and_monitor(self):
        try:
            self.jobs = []
            inputs = self.collect_common_inputs()
            if not inputs: 
                self.stop_monitoring()
                return

            self.display_output_message("--- Generating Macro Files ---")
            for key, val in inputs.items():
                if isinstance(val, pd.DataFrame):
                    self.append_output_message(f"{key}:\n{val.to_string()}")
                else:
                    self.append_output_message(f"{key}: {val}")
            
            if "Tile Scan" in self.workflow_mode_var.get():
                self.generate_tile_scan_macros(inputs)
            else:
                self.generate_multipoint_macros(inputs)

            if not self.jobs:
                 self.append_output_message("No macro jobs were created. Stopping.")
                 self.stop_monitoring()
                 return

            first_macro = self.jobs[0]['dapi_macro_path']
            run_command = f'RunMacro("{first_macro}");'
            self.display_output_message(f"MACROS GENERATED. Start acquisition in Nikon with:\n\n{run_command}")
            self.append_output_message("\n--- Starting Real-Time Monitoring ---")

            await self.monitor_acquisition_progress(inputs)

        except Exception as e:
            self.append_output_message(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_monitoring()

    def write_macro_pair(self, f_dapi, f_acq, params):
        if params['is_first']:
            f_dapi.write(f"// Date: {date.today().strftime('%b-%d-%Y')}\n")
            f_dapi.write(f"// Z Stack = {params['z_step_num']} slices, Z step size = {params['z_step_size']} um\n\n")
            f_dapi.write('ShowImageInfoBeforeSaveAs(0);\nLUTs_KeepAutoScale(1);\n')
            f_dapi.write(f'ND_LoadExperiment("{params["xml_file"]}");\n\n')

        for _, row in params['channels_df'].iterrows():
            f_dapi.write(f'SelectOptConf("{row["Filter"]}");\nCameraSet_Exposure(1, {row["Exposure"]});\n')
        f_dapi.write('Stg_SetShutterStateEx("EPI", 0);\n')
        f_dapi.write(params['move_cmd'])
        f_dapi.write(f"StgMoveMainZ({params['z']:.4f}, 0);\n")
        f_dapi.write('Stg_SetShutterStateEx("EPI", 1);\n')
        f_dapi.write(f'SelectOptConf("{params["focus_chan"]}");\nLiveSync();\n')
        nd2_dapi_path = Path(params['img_save_path']) / f"{params['file_prefix']}_{params['focus_chan']}_zScan.nd2"
        f_dapi.write(f'ND_DefineExperiment(0, 0, 1, 0, 1, "{nd2_dapi_path}","",0,0,0,0);\n')
        f_dapi.write(f'ND_SetZSeriesExp(2, 0, {params["z"]:.4f}, 0, {params["focus_step"]}, {params["focus_count"]}, 0, 0, "", "", "");\n')
        f_dapi.write('ND_RunExperiment(0);\nCloseCurrentDocument(0);\n\n')
        f_dapi.write('Stg_SetShutterStateEx("EPI", 0);\n')
        f_dapi.write(f'WaitText({params["wait_time"] * 60}, "Waiting for Python to compute focus plane...");\n')
        f_dapi.write(f'RunMacro("{params["acq_macro_path"]}");\n')

        for _, row in params['channels_df'].iterrows():
            f_acq.write(f'SelectOptConf("{row["Filter"]}");\nCameraSet_Exposure(1, {row["Exposure"]});\n')
        f_acq.write('Stg_SetShutterStateEx("EPI", 0);\n')
        f_acq.write(f"StgMoveMainZ({params['z']:.4f}, 0);\n")
        f_acq.write('Stg_SetShutterStateEx("EPI", 1);\n')
        f_acq.write(f'SelectOptConf("{params["focus_chan"]}");\nLiveSync();\n')
        nd2_acq_path = Path(params['img_save_path']) / f"{params['file_prefix']}_chStack.nd2"
        f_acq.write(f'ND_DefineExperiment(0, 0, 1, 1, 1, "{nd2_acq_path}","",0,0,0,0);\n')
        f_acq.write(f'ND_SetZSeriesExp(2, 0, {params["z"]:.4f}, 0, {params["z_step_size"]}, {params["z_step_num"]}, 0, 0, "", "", "");\n')
        f_acq.write('ND_RunExperiment(0);\nCloseCurrentDocument(0);\n\n')

    def finalize_and_chain_macros(self):
        if not self.jobs: return
        for i in range(len(self.jobs) - 1):
            with open(self.jobs[i]['acq_macro_path'], "a") as f:
                f.write(f'RunMacro("{self.jobs[i+1]["dapi_macro_path"]}");\n')
        with open(self.jobs[-1]['acq_macro_path'], "a") as f:
            f.write('Stg_Light_SetIrisIntensity("EPI", 0);\n')
            f.write('StgMoveMainZ(-4000, 1); // final\n')

    def generate_tile_scan_macros(self, inputs):
        self.append_output_message("... Running Tile Scan macro generation ...")
        pts = self.get_pts_from_xml(inputs['xml_file'])
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
                move_cmd = f"StgMoveXY({xCurr:.4f}, {yCurr:.4f}, 0);\n" if (ii == 0 and jj == 0) else f"StgMoveXY({(step_dist_X * xScanDir if jj!=0 else 0):.4f}, {(step_dist_Y if jj==0 else 0):.4f}, 1);\n"
                initial_z = a * xCurr + b * yCurr + c
                params = {**inputs, 'z': initial_z, 'is_first': (ii == 0 and jj == 0), 'file_prefix': file_prefix, 'acq_macro_path': acq_macro_path, 'move_cmd': move_cmd}
                self.jobs.append({'dapi_macro_path': dapi_macro_path, 'acq_macro_path': acq_macro_path, 'dapi_nd2_prefix': file_prefix, 'original_z': initial_z})
                with open(dapi_macro_path, "w") as f_dapi, open(acq_macro_path, "w") as f_acq:
                    self.write_macro_pair(f_dapi, f_acq, params)
        self.finalize_and_chain_macros()
        self.append_output_message(f"Generated {len(self.jobs)} macro pairs.")

    def generate_multipoint_macros(self, inputs):
        self.append_output_message("... Running Multipoint macro generation ...")
        pts = self.get_pts_from_xml(inputs['xml_file'])
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
                    params = {**inputs, 'z': pz, 'is_first': (macro_file_count == 2), 'file_prefix': file_prefix, 'acq_macro_path': acq_macro_path, 'move_cmd': f"StgMoveXY({xCurr:.4f}, {yCurr:.4f}, 0);\n"}
                    self.jobs.append({'dapi_macro_path': dapi_macro_path, 'acq_macro_path': acq_macro_path, 'dapi_nd2_prefix': file_prefix, 'original_z': pz})
                    with open(dapi_macro_path, "w") as f_dapi, open(acq_macro_path, "w") as f_acq:
                        self.write_macro_pair(f_dapi, f_acq, params)
        self.finalize_and_chain_macros()
        self.append_output_message(f"Generated {len(self.jobs)} macro pairs.")

    def get_pts_from_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        x_coords = [eval(elem.attrib['value']) for elem in root.findall('.//dPosX')[0]]
        y_coords = [eval(elem.attrib['value']) for elem in root.findall('.//dPosY')[0]]
        z_coords = [eval(elem.attrib['value']) for elem in root.findall('.//dPosZ')[0]]
        return np.stack([x_coords, y_coords, z_coords], axis=1)

    def update_macro_with_new_focus(self, macro_path, new_z_focus, z_step_size, z_step_num):
        try:
            with open(macro_path, 'r') as file: content = file.readlines()
            with open(macro_path, "w") as macro_file:
                for line in content:
                    if 'ND_SetZSeriesExp' in line:
                        macro_file.write(f'ND_SetZSeriesExp(2, 0, {new_z_focus:.4f}, 0, {z_step_size}, {z_step_num}, 0, 0, "", "", "");\n')
                    elif 'StgMoveMainZ' in line and 'final' not in line:
                        macro_file.write(f"StgMoveMainZ({new_z_focus:.4f}, 0);\n")
                    else:
                        macro_file.write(line)
        except Exception as e:
            self.append_output_message(f"ERROR: Could not update macro {macro_path.name}: {e}")

    async def monitor_acquisition_progress(self, inputs):
        img_save_path = inputs['img_save_path']
        focus_chan = inputs['focus_chan']
        focus_step = inputs['focus_step']
        num_tiles = len(self.jobs)
        
        for i, job in enumerate(self.jobs):
            if self.stop_monitoring_event.is_set():
                self.append_output_message("Monitoring stopped by user."); break
            
            progress = f"({i+1}/{num_tiles})"
            dapi_file_prefix = job['dapi_nd2_prefix']
            self.append_output_message(f"\n{progress} Waiting for autofocus file: {dapi_file_prefix}...")

            dapi_nd2_file = None
            start_wait_time = time.time()
            while not self.stop_monitoring_event.is_set():
                found_files = list(img_save_path.glob(f"{dapi_file_prefix}*{focus_chan}*zScan.nd2"))
                if found_files:
                    try:
                        with nd2.ND2File(str(found_files[0])) as ndfile:
                            if ndfile.is_legacy: # Check for file integrity
                                 pass 
                        dapi_nd2_file = found_files[0]
                        self.append_output_message(f"Found: {dapi_nd2_file.name}"); break
                    except Exception: await asyncio.sleep(2)
                else: await asyncio.sleep(2)
                if time.time() - start_wait_time > 600:
                     self.append_output_message("CRASH DETECTED: Timeout waiting for file."); return

            if not dapi_nd2_file: continue

            self.append_output_message(f"{progress} Computing focus plane...")
            
            # --- MODIFIED: Use nd2.imread for robust dask array creation ---
            img_stack_dask = nd2.imread(str(dapi_nd2_file), dask=True)
            if img_stack_dask.ndim == 4 and img_stack_dask.shape[1] == 1: # Handle ZCYX format
                img_stack_dask = img_stack_dask[:, 0, :, :] # Convert to ZYX

            idxFocus = self.findFocusLapVar(img_stack_dask)
            num_z_planes = img_stack_dask.shape[0]
            
            original_z = job['original_z']
            z_range_total = focus_step * num_z_planes
            z_range_um = np.linspace(original_z - z_range_total / 2, original_z + z_range_total / 2, num_z_planes)
            zFocus = z_range_um[idxFocus]

            self.append_output_message(f"----------------------------------------")
            self.append_output_message(f"  Computed Focus: Plane {idxFocus + 1} / {num_z_planes}")
            self.append_output_message(f"  Original Z: {original_z:.2f} µm  ->  New Z: {zFocus:.2f} µm")
            self.append_output_message(f"----------------------------------------")

            acq_macro_path = job['acq_macro_path']
            self.append_output_message(f"{progress} Updating macro: {acq_macro_path.name}")
            self.update_macro_with_new_focus(acq_macro_path, zFocus, inputs['z_step_size'], inputs['z_step_num'])
            
            if i > 0:
                elapsed_time = time.time() - self.start_time
                avg_time_per_tile = elapsed_time / i
                remaining_tiles = num_tiles - (i + 1)
                remaining_time_sec = remaining_tiles * avg_time_per_tile
                remaining_time_min = remaining_time_sec / 60
                self.append_output_message(f"Estimated time remaining: {remaining_time_min:.1f} minutes")

    def collect_common_inputs(self):
        if not self.base_path_var.get() or not self.xml_file_var.get():
            messagebox.showerror("Error", "Please select an experiment XML file first."); return None
        inputs = {}
        base_path = Path(self.base_path_var.get())
        inputs['xml_file'] = Path(self.xml_file_var.get())
        if not base_path.exists() or not inputs['xml_file'].exists(): raise ValueError("Base Path or XML File does not exist.")
        e_drive = Path('E:/')
        run_path = base_path
        if e_drive.exists():
            local_path = e_drive / ' '.join(base_path.parts[-2:])
            run_path = local_path
            if base_path.resolve() != run_path.resolve():
                run_path.mkdir(parents=True, exist_ok=True)
                try: shutil.copytree(base_path, run_path, dirs_exist_ok=True)
                except Exception as e: print(f"Could not copy from base to local path: {e}")
        inputs['base_path'] = base_path; inputs['run_path'] = run_path
        inputs['z_step_num'] = int(self.z_step_num_var.get()); inputs['z_step_size'] = float(self.z_step_size_var.get())
        inputs['focus_count'] = int(self.focus_count_var.get()); inputs['focus_step'] = float(self.focus_step_var.get())
        inputs['focus_chan'] = self.focus_chan_var.get(); inputs['wait_time'] = float(self.wait_time_var.get())
        inputs['objective_str'] = self.objective_var.get(); inputs['overlap_frac'] = float(self.overlap_frac_var.get())
        if "Multipoint" in self.workflow_mode_var.get():
            inputs['tiles_x'] = int(self.tiles_x_var.get()); inputs['tiles_y'] = int(self.tiles_y_var.get())
        channels_df = pd.DataFrame()
        channels_df['Filter'] = self.channel_vars.keys()
        exposures = [int(self.channel_vars[ch].get()) for ch in channels_df['Filter']]
        channels_df['Exposure'] = exposures
        inputs['channels_df'] = channels_df.loc[channels_df['Exposure'] > 0].reset_index(drop=True)
        if inputs['focus_chan'] not in inputs['channels_df']['Filter'].values:
            raise ValueError(f"Focus channel '{inputs['focus_chan']}' not in active channels.")
        all_res = {'20X': 0.325, '40X (Air)': 0.1663, '40X (Oil)': 0.1625, '60X': 0.1083}
        if inputs['objective_str'] not in all_res: raise ValueError("Invalid objective selected.")
        inputs['img_res'] = all_res[inputs['objective_str']]
        inputs['macro_save_path'] = run_path / 'Macros'
        inputs['img_save_path'] = run_path / '00 RAW'
        inputs['macro_save_path'].mkdir(exist_ok=True); inputs['img_save_path'].mkdir(exist_ok=True)
        for f in inputs['macro_save_path'].glob('*.mac'): os.remove(f)
        self.start_time = time.time()
        return inputs

if __name__ == "__main__":
    app = NikonMacroGUI()
    app.mainloop()

