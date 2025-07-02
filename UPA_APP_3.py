import sys
import os



import pandas as pd
import numpy as np
import datetime # Use 'import datetime'
import time
import warnings
from tqdm import tqdm
import concurrent.futures
from collections import Counter

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

import cx_Oracle
import teradatasql
from sqlalchemy import create_engine # Añadir esta importación

# import teradatasql # Kept commented as in original

# Define the pozo class at the top level
class pozo:
    def __init__(self, nombre, sistema_extraccion, controles, declino, cabeza_de_pozo, instalaciones, fecha_PEM, interferencias, perfil_presiones, downtime,area,fecha_ultima_UPA_data,fecha_UPA_actual_data,diagnostico_data, UPA_actual_df_for_pozo, ULTIMA_UPA_df_for_pozo):
        self.nombre = nombre
        self.sistema_extraccion = sistema_extraccion
        self.controles = controles
        self.declino = declino
        self.cabeza_de_pozo = cabeza_de_pozo
        self.instalaciones = instalaciones
        self.fecha_PEM = fecha_PEM
        self.interferencias = interferencias # This is a DataFrame filtered for the specific pozo
        self.perfil_presiones = perfil_presiones
        self.downtime = downtime
        self.area = area
        self.diagnostico_data = diagnostico_data # This is a DataFrame/Series filtered for the specific pozo

        self.actividad = None
        self.corte_produccion = None
        self.fecha_capex = None
        self.bruta_declino_inicial = None
        self.interferencias_previas = None
        self.meses_desplazados = 0 # Initialize
        self.fecha_upa_anterior = None
        self.fecha_crono = None
        self.presion_cabeza_actual = None
        self.presion_linea = None
        self.relacion_presion_linea = None
        self.ultimo_control = None
        self.fecha_ultimo_control = None
        self.campana = None
        self.estado_actividad= None
        self.fecha_capex_ajustada = None
        self.orificio_ult_control = None
        self.estado_operativo=None
        self.ultimo_paro_rubro=None
        self.ultimo_paro_fecha=None
        self.tiempo_ultimo_paro=None
        self.ultimo_paro_gran_rubro=None
        self.correccion= None
        self.prioridad = None
        self.rodlock = None
        self.fecha_rodlock= None
        self.asegurado_armadura=None
        self.fecha_UPA_actual= None
        self.tiempo_cierre_actividad=None # This will be a Series or single value
        self.actividad_UPA=None

        # Process diagnostico
        if not self.diagnostico_data.empty:
            # Assuming diagnostico_data is already filtered for the current pozo
            # and sorted if multiple entries exist for the same pozo (taking the last one)
            self.diagnostico = self.diagnostico_data # Store the pre-filtered diagnostic data
            # If diagnostico_data is a DataFrame with multiple rows for the pozo, you might need:
            # self.diagnostico = self.diagnostico_data.sort_values('FECHA_DIAGNOSTICO', ascending=False).iloc[0] if not self.diagnostico_data.empty else pd.Series()

        if not self.perfil_presiones.empty:
            if 'FECHA' in self.perfil_presiones.columns and not self.perfil_presiones['FECHA'].empty:
                latest_pressure_idx = self.perfil_presiones['FECHA'].idxmax()
                self.presion_cabeza_actual = self.perfil_presiones.loc[latest_pressure_idx, 'PRESION_CABEZA']
                self.presion_linea = self.perfil_presiones.loc[latest_pressure_idx, 'PRESION_LINEA']
                if pd.notna(self.presion_linea) and self.presion_linea != 0 and pd.notna(self.presion_cabeza_actual):
                    self.relacion_presion_linea = self.presion_cabeza_actual / self.presion_linea
                else:
                    self.relacion_presion_linea = None


        if not self.interferencias.empty:
            if 'Inicio Fractura' in self.interferencias.columns and not self.interferencias['Inicio Fractura'].empty:
                # Ensure 'Inicio Fractura' is datetime
                self.interferencias['Inicio Fractura'] = pd.to_datetime(self.interferencias['Inicio Fractura'], errors='coerce')
                self.interferencias['fecha_interferencia'] = self.interferencias['Inicio Fractura'].dt.to_period('M').dt.start_time


        if not self.controles.empty:
            controles_A = self.controles[self.controles['TEST_PURP_CD'] == 'A']
            if 'TEST_DT' in controles_A.columns and not controles_A['TEST_DT'].empty:
                # Ensure TEST_DT is datetime before finding idxmax
                controles_A['TEST_DT'] = pd.to_datetime(controles_A['TEST_DT'], errors='coerce')
                if not controles_A['TEST_DT'].dropna().empty:
                    ultimo_control_idx = controles_A['TEST_DT'].idxmax()
                    if pd.notna(ultimo_control_idx):
                        self.ultimo_control = controles_A.loc[ultimo_control_idx, 'BRUTA']
                        self.fecha_ultimo_control = controles_A.loc[ultimo_control_idx, 'TEST_DT']
                        self.orificio_ult_control = controles_A.loc[ultimo_control_idx, 'CHOKE_SIZE']
                   
        if sistema_extraccion == 'FA':
            componentes_pozo = self.instalaciones # Already filtered for the pozo
            if not componentes_pozo.empty and 'COMPONENTE' in componentes_pozo.columns and \
               componentes_pozo['COMPONENTE'].astype(str).str.contains('VARILLA BOMBEO', case=False, na=False).any():
                # print(f"El pozo {self.nombre} tiene 'VARILLA BOMBEO', no se asigna actividad BIF.")
                self.actividad = None # Or some other indicator
            else:
                self.actividad = 'BIF'
                if pd.notna(self.ultimo_control):
                    if 80 <= self.ultimo_control <= 110:
                        self.estado_actividad = 'En ventana BIF'
                    elif self.ultimo_control < 80:
                        self.estado_actividad = 'Atrasada BIF'
                    elif self.ultimo_control > 110:
                        self.estado_actividad = 'Fuera de ventana BIF'
                self.corte_produccion = 110
        elif sistema_extraccion == 'FL':
            self.actividad = 'CBM'
            if pd.notna(self.ultimo_control):
                if 25 <= self.ultimo_control <= 35:
                    self.estado_actividad = 'En ventana CBM'
                elif 15 <= self.ultimo_control < 25:
                    self.estado_actividad = 'Atrasada CBM'
                elif self.ultimo_control > 35 or self.ultimo_control < 15 : # Adjusted logic
                    self.estado_actividad = 'Fuera de ventana CBM'
            self.corte_produccion = 35
        # else:
            # print(f"Método de producción no es FA o FL para {self.nombre} - {sistema_extraccion}")

        if not self.declino.empty:
             self.declino['Declino_pasado'] = None # This seems to be an unused column initialization
        
        if pd.notna(self.fecha_PEM):
            self.campana = pd.Timestamp(self.fecha_PEM).year # Use .year for integer year

        if not self.downtime.empty:
            self.downtime['PROD_DT'] = pd.to_datetime(self.downtime['PROD_DT'], errors='coerce')
            if not self.downtime['PROD_DT'].dropna().empty:
                latest_downtime_idx = self.downtime['PROD_DT'].idxmax()
                self.ultimo_paro_rubro = str(self.downtime.loc[latest_downtime_idx, 'RUBRO'])
                self.ultimo_paro_fecha = self.downtime.loc[latest_downtime_idx, 'PROD_DT']
                self.tiempo_ultimo_paro = self.downtime.loc[latest_downtime_idx, 'HORAS_DE_PARO'] 
                self.ultimo_paro_gran_rubro = str(self.downtime.loc[latest_downtime_idx, 'GRAN_RUBRO'])
                
                if pd.notna(self.tiempo_ultimo_paro) and self.tiempo_ultimo_paro >= 24:
                    if 'Coiled' in self.ultimo_paro_rubro or 'CTU' in self.ultimo_paro_rubro:
                        self.estado_operativo = 'Arenado o Espera de CTU'
                    elif 'Tractor' in self.ultimo_paro_rubro or 'TRACTOR' in self.ultimo_paro_gran_rubro:
                        self.estado_operativo = 'En falla o siendo intervenido'
                elif pd.notna(self.tiempo_ultimo_paro) and self.tiempo_ultimo_paro < 24:
                    self.estado_operativo = 'Operativo'
        
        if not fecha_ultima_UPA_data.empty:
            self.fecha_upa_anterior = pd.Timestamp(fecha_ultima_UPA_data.iloc[0]) if not fecha_ultima_UPA_data.empty else pd.NaT
        
        if not fecha_UPA_actual_data.empty:
            self.fecha_UPA_actual = pd.Timestamp(fecha_UPA_actual_data.iloc[0]) if not fecha_UPA_actual_data.empty else pd.NaT
            # Get 'Días de Cierre' from UPA_actual_df_for_pozo for the current pozo
            if not UPA_actual_df_for_pozo.empty:
                pozo_data_in_upa_actual = UPA_actual_df_for_pozo[UPA_actual_df_for_pozo['NOMBRE POZO'] == self.nombre]
                if not pozo_data_in_upa_actual.empty and 'Días de Cierre' in pozo_data_in_upa_actual.columns:
                     self.tiempo_cierre_actividad = pozo_data_in_upa_actual['Días de Cierre'].iloc[0]
                     self.actividad_UPA = pozo_data_in_upa_actual['EVENTOS ASOCIADOS AL CIERRE'].iloc[0]



        if isinstance(self.cabeza_de_pozo, pd.DataFrame) and not self.cabeza_de_pozo.empty and 'NOMBRE_COMP' in self.cabeza_de_pozo.columns:
            rod_lock_bop_filter = self.cabeza_de_pozo['NOMBRE_COMP'].astype(str).str.contains('ROD LOCK BOP', na=False)
            self.rodlock = self.cabeza_de_pozo[rod_lock_bop_filter].shape[0]
            if self.rodlock > 0:
                self.cabeza_de_pozo['FECHA_INSTALACION'] = pd.to_datetime(self.cabeza_de_pozo['FECHA_INSTALACION'], errors='coerce')
                self.fecha_rodlock = self.cabeza_de_pozo[rod_lock_bop_filter]['FECHA_INSTALACION'].max()
                self.fecha_rodlock = pd.Timestamp(self.fecha_rodlock) if pd.notna(self.fecha_rodlock) else pd.NaT
            else:
                # Check for VALVULA MAESTRA and VARILLA BOMBEO
                if not self.instalaciones.empty and 'COMPONENTE' in self.instalaciones.columns and \
                   not self.cabeza_de_pozo.empty and 'FECHA_INSTALACION' in self.cabeza_de_pozo.columns:
                    
                    self.cabeza_de_pozo['FECHA_INSTALACION'] = pd.to_datetime(self.cabeza_de_pozo['FECHA_INSTALACION'], errors='coerce')
                    ultima_fecha_instalacion_cabeza = self.cabeza_de_pozo['FECHA_INSTALACION'].max()

                    vm_filter = (self.cabeza_de_pozo['NOMBRE_COMP'].astype(str).str.contains('VALVULA MAESTRA', na=False)) & \
                                (self.cabeza_de_pozo['FECHA_INSTALACION'] == ultima_fecha_instalacion_cabeza)
                    VM_count = self.cabeza_de_pozo[vm_filter].shape[0]
                    
                    VB_present = self.instalaciones['COMPONENTE'].astype(str).str.contains('VARILLA BOMBEO', case=False, na=False).any()

                    if VM_count > 0 and VB_present:
                        self.asegurado_armadura = "Asegurado con Armadura y varillas en pesca"

    def calcular_fecha_capex(self):
        if self.declino.empty or 'BRUTA_(m3/DC)' not in self.declino.columns or self.corte_produccion is None:
            return

        # Ensure FECHA is datetime
        self.declino['FECHA'] = pd.to_datetime(self.declino['FECHA'], errors='coerce')
        declino_sorted = self.declino.sort_values('FECHA')
        
        self.bruta_declino_inicial = None

        if declino_sorted['BRUTA_(m3/DC)'].isnull().all():
            return

        # Find the point of maximum production to start declino analysis from there
        if not declino_sorted['BRUTA_(m3/DC)'].dropna().empty:
            max_prod_idx = declino_sorted['BRUTA_(m3/DC)'].idxmax()
            declino_analysis_start_date = declino_sorted.loc[max_prod_idx, 'FECHA']
            declino_from_max = declino_sorted[declino_sorted['FECHA'] >= declino_analysis_start_date]
        else:
            declino_from_max = declino_sorted # Or handle as no valid declino

        for _, row in declino_from_max.dropna(subset=['BRUTA_(m3/DC)']).iterrows():
            if pd.notna(row['BRUTA_(m3/DC)']) and row['BRUTA_(m3/DC)'] <= self.corte_produccion:
                self.fecha_capex = row['FECHA']
                self.fecha_capex_ajustada = row['FECHA'] # Initial adjustment
                self.bruta_declino_inicial = row['BRUTA_(m3/DC)']
                self.correccion="Sin correcion - Fecha declino"
                break
 
        if self.interferencias is not None and not self.interferencias.empty and pd.notna(self.fecha_capex) and 'fecha_interferencia' in self.interferencias.columns:
            # Ensure 'Meses_Desplazados' is numeric and integer
            self.interferencias['Meses_Desplazados'] = pd.to_numeric(self.interferencias['Meses_Desplazados'], errors='coerce').fillna(0).astype(int)

            interferencias_previas_df = self.interferencias[self.interferencias['fecha_interferencia'] <= self.fecha_capex]
            meses_desplazados_prev = interferencias_previas_df['Meses_Desplazados'].sum()
            
            self.meses_desplazados = meses_desplazados_prev # Store sum of previous displacements
            if pd.notna(self.fecha_capex_ajustada) and meses_desplazados_prev > 0:
                 self.fecha_capex_ajustada += pd.DateOffset(months=int(meses_desplazados_prev))
                 self.correccion="Con correcion - Interferencias previas al declino"

            # Consider interferences up to 1 month after initial capex date for further adjustment
            fecha_capex_mas_1_mes = self.fecha_capex + pd.DateOffset(months=1) # Changed from 2 to 1 as per logic in verificar_interferencias
            interferencias_post_1_mes = self.interferencias[
                (self.interferencias['fecha_interferencia'] > self.fecha_capex) &
                (self.interferencias['fecha_interferencia'] <= fecha_capex_mas_1_mes)
            ]
            meses_desplazados_post_1 = interferencias_post_1_mes['Meses_Desplazados'].sum()
            
            if pd.notna(self.fecha_capex_ajustada) and meses_desplazados_post_1 > 0:
                self.fecha_capex_ajustada += pd.DateOffset(months=int(meses_desplazados_post_1))
                self.meses_desplazados += meses_desplazados_post_1 # Add to total
                # Update correccion if it was already set
                if "Con correcion" in str(self.correccion):
                    self.correccion += " e interferencias post-declino (+1 mes)"
                else:
                    self.correccion = "Con correcion - Interferencias post-declino (+1 mes)"


    def set_fecha_upa_anterior(self, fecha_upa_anterior): # This method seems unused if data passed in init
        self.fecha_upa_anterior = fecha_upa_anterior

    def set_fecha_crono(self, fecha_crono):
        self.fecha_crono = fecha_crono # fecha_crono should be a single Timestamp or NaT

    def set_fecha_upa_actual(self, fecha_upa_actual): # This method seems unused
        self.fecha_upa_actual = fecha_upa_actual

    def verificar_interferencias(self, fecha_base_ajuste):
        if self.interferencias is None or self.interferencias.empty or pd.isna(fecha_base_ajuste) or 'fecha_interferencia' not in self.interferencias.columns:
            return fecha_base_ajuste

        current_adjusted_date = pd.Timestamp(fecha_base_ajuste)
        initial_total_displacement_for_verification = 0
        
        # Ensure 'Meses_Desplazados' is numeric and integer
        self.interferencias['Meses_Desplazados'] = pd.to_numeric(self.interferencias['Meses_Desplazados'], errors='coerce').fillna(0).astype(int)

        correcciones_count = 0
        max_correcciones = 10

        while correcciones_count < max_correcciones:
            made_adjustment_this_loop = False
            
            # Check month before, current month, and month after current_adjusted_date
            # Convert current_adjusted_date to period for month comparison
            current_month_period = current_adjusted_date.to_period('M')
            
            # Interferencias en el mes anterior al mes de current_adjusted_date
            prev_month_interferences = self.interferencias[
                self.interferencias['fecha_interferencia'].dt.to_period('M') == (current_month_period - 1)
            ]
            # Interferencias en el mes de current_adjusted_date
            current_month_interferences = self.interferencias[
                self.interferencias['fecha_interferencia'].dt.to_period('M') == current_month_period
            ]
            # Interferencias en el mes siguiente al mes de current_adjusted_date, con MPE_rem > 50
            next_month_interferences = self.interferencias[
                (self.interferencias['fecha_interferencia'].dt.to_period('M') == (current_month_period + 1)) &
                (pd.to_numeric(self.interferencias.get('MPE_rem', 0), errors='coerce').fillna(0) > 50) # Added .get for safety
            ]

            total_displacement_this_check = 0
            
            # Logic from original: if prev_month has displacement, it seems to pull date earlier
            # This logic might need review based on desired outcome.
            # The original code had: fecha_capex -= pd.DateOffset(months=meses_desplazados_anterior_mes-1)
            # which is unusual. Assuming it means "if there was an interference that should have displaced
            # something scheduled for *this* month, but the interference was last month, adjust"
            # For now, let's sum displacements that affect the *current* scheduling window.

            # Re-interpreting the original logic:
            # It checks 3 windows relative to current_adjusted_date's month.
            
            # Displacement from previous month's interferences
            disp_prev = prev_month_interferences['Meses_Desplazados'].sum()
            # Displacement from current month's interferences
            disp_curr = current_month_interferences['Meses_Desplazados'].sum()
            # Displacement from next month's (MPE > 50) interferences
            disp_next = next_month_interferences['Meses_Desplazados'].sum()

            if disp_prev == 0 and disp_curr == 0 and disp_next == 0:
                break # No relevant interferences in the window

            # Original logic for adjusting:
            # This part is tricky. The original code had a specific order.
            # If prev month had displacement: current_adjusted_date -= pd.DateOffset(months=disp_prev - 1)
            # This is very specific and might be business logic. Let's try to replicate.
            # If disp_prev > 0, it means an interference in the month *before* current_adjusted_date's month
            # caused a displacement. The "-1" is confusing.
            # Let's assume the goal is to push current_adjusted_date if there's an overlapping interference.

            # Simplified: if any interference in these windows causes displacement, add it.
            # This might lead to over-correction if not careful.
            # The original loop implies iterative adjustment.

            # Let's try to apply the sum of displacements from interferences *at or before* the current_adjusted_date's month
            # that would push it out.
            
            effective_displacement_this_iteration = 0
            if disp_curr > 0:
                effective_displacement_this_iteration = disp_curr
                current_adjusted_date += pd.DateOffset(months=int(effective_displacement_this_iteration))
                initial_total_displacement_for_verification += effective_displacement_this_iteration
                made_adjustment_this_loop = True
            elif disp_next > 0: # Only if current month had no displacement
                effective_displacement_this_iteration = disp_next
                current_adjusted_date += pd.DateOffset(months=int(effective_displacement_this_iteration))
                initial_total_displacement_for_verification += effective_displacement_this_iteration
                made_adjustment_this_loop = True
            # The prev_month logic is harder to fit cleanly without more context on the "-1".
            # For now, focusing on current and future pushes.

            if not made_adjustment_this_loop:
                 break # No adjustments made, exit loop

            correcciones_count += 1
            if current_adjusted_date.year > 2025: # Or a relevant future year limit
                # print(f"Advertencia: La fecha ajustada {current_adjusted_date} para {self.nombre} se salió del rango de planificación.")
                break
        
        if correcciones_count == max_correcciones:
            # print(f"Advertencia: Se alcanzó el máximo de correcciones ({max_correcciones}) para {self.nombre} en verificar_interferencias.")
            pass

        if pd.notna(self.fecha_capex_ajustada): # Check if it was set before
            self.fecha_capex_ajustada = current_adjusted_date
            if initial_total_displacement_for_verification > 0 : # Only update if displacement occurred
                self.meses_desplazados = (self.meses_desplazados or 0) + initial_total_displacement_for_verification
                self.correccion = f"Ajustado por verificar_interferencias, correcciones: {correcciones_count}"
            
        return current_adjusted_date


    def set_prioridad(self, prioridad):
        self.prioridad = prioridad

    def search_declino(self, fecha_busqueda):
        if self.declino is None or self.declino.empty or 'FECHA' not in self.declino.columns or 'PETRÓLEO_(m3/DC)' not in self.declino.columns:
            return None
        
        # Ensure fecha_busqueda is a Timestamp for comparison
        fecha_busqueda_ts = pd.Timestamp(fecha_busqueda)
        
        # Find the closest date in declino (on or before fecha_busqueda_ts)
        # self.declino['FECHA'] should already be datetime
        relevant_declino = self.declino[self.declino['FECHA'] <= fecha_busqueda_ts]
        if not relevant_declino.empty:
            closest_date_row = relevant_declino.loc[relevant_declino['FECHA'].idxmax()]
            return closest_date_row['PETRÓLEO_(m3/DC)']
        return None


class UPAWorkflow:
    def __init__(self):
        self.console = Console()
        # Initialize DataFrames
        self.UPS_DIM_COMPLETACION = pd.DataFrame()
        self.CNS_NOC_PI = pd.DataFrame()
        self.CNS_NOC_TOW_CONTROLES = pd.DataFrame()
        self.CNS_NOC_TOW_PAR_PERD = pd.DataFrame()
        self.NOC_GR_PERFIL_UPA_DECLINO = pd.DataFrame()
        self.GIDI_POZO = pd.DataFrame()
        self.PA_2025_Activos = pd.DataFrame()
        self.UPS_FT_PROY_CONSULTA_ACTIVIDAD = pd.DataFrame()
        self.FDD_CNS_NOC_OW_INSTALACIONES = pd.DataFrame()
        self.UPS_FT_CABEZA_POZO = pd.DataFrame()
        self.ULTIMA_UPA = pd.DataFrame()
        self.UPA_actual = pd.DataFrame()
        self.FDD_CNS_GRALO_FDP_DIAGNOSTICO = pd.DataFrame() # Renamed from FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos for consistency
        self.FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos = pd.DataFrame()


        # Derived DataFrames / lists
        self.UPS_FT_CABEZA_POZO_filtrado = pd.DataFrame()
        self.actividad_aseguramiento = pd.DataFrame()
        self.FDD_CNS_NOC_OW_INSTALACIONES_ultimos = pd.DataFrame()
        self.lista_pozos = []
        self.df_upa_sin_limite = pd.DataFrame()
        self.df_upa_con_limite = pd.DataFrame()


    def _normalize_well_name(self, name):
        if not isinstance(name, str):
            return ""
        normalized = name.lower().strip()
        if normalized.startswith('ypf.nq.'):
            normalized = normalized[len('ypf.nq.'):]
        if normalized.endswith('(h)'):
            normalized = normalized[:-3]
        # Puedes añadir más reglas de normalización si es necesario
        # Por ejemplo, para quitar espacios internos o guiones:
        # normalized = normalized.replace('-', '').replace(' ', '')
        return normalized

    def _read_and_process_excel(self, file_path, columns_to_read_config):
        # start_time = time.time() # tqdm handles progress
        try:
            usecols = columns_to_read_config.get(file_path, None)
            df = pd.read_excel(file_path, usecols=usecols)
        except Exception as e:
            self.console.print(f"[red]Error reading {file_path}: {e}[/red]")
            return file_path, pd.DataFrame()  # Return empty DataFrame on error 
        
        if 'NOMBRE_POZO' in df.columns:
            df['NOMBRE_POZO'] = df['NOMBRE_POZO'].astype(str).str.replace('YPF.Nq.', '', regex=False)
        if 'PETRÓLEO_(m3/DC)' in df.columns and 'AGUA_(m3/DC)' in df.columns: # Ensure AGUA column exists
            df['PETRÓLEO_(m3/DC)'] = pd.to_numeric(df['PETRÓLEO_(m3/DC)'], errors='coerce').fillna(0)
            df['AGUA_(m3/DC)'] = pd.to_numeric(df['AGUA_(m3/DC)'], errors='coerce').fillna(0)
            df['BRUTA_(m3/DC)'] = df['PETRÓLEO_(m3/DC)'] + df['AGUA_(m3/DC)']
        elif 'PETRÓLEO_(m3/DC)' in df.columns: # Handle if only PETROLEO exists
             df['PETRÓLEO_(m3/DC)'] = pd.to_numeric(df['PETRÓLEO_(m3/DC)'], errors='coerce').fillna(0)
             # Assuming BRUTA should be PETROLEO if AGUA is missing
             df['BRUTA_(m3/DC)'] = df['PETRÓLEO_(m3/DC)']


        if 'HORAS_DE_PARO' in df.columns:
            df['HORAS_DE_PARO'] = pd.to_numeric(df['HORAS_DE_PARO'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
        # end_time = time.time()
        # self.console.print(f"Processed {file_path} in {end_time - start_time:.2f} seconds")
        return file_path, df  # Return the original file_path and the DataFrame

    def run_section_1_load_excel_data(self):
        self.console.rule("[bold blue]1. Carga y Procesamiento Inicial de Datos Excel[/bold blue]")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                columns_to_read_config = {
                    "datos_prueba/UPS_DIM_COMPLETACION.xlsx": ['Completacion_Nombre_Corto', 'Metodo_Produccion_Actual_Cd', 'Bloque_Monitoreo_Nombre', 'Fecha_Inicio_Produccion_Dt'],
                    "datos_prueba/CNS_NOC_PI.xlsx": ['NOMBRE_CORTO_POZO', 'FECHA', 'PRESION_CABEZA', 'PRESION_LINEA'],
                    "datos_prueba/CNS_NOC_TOW_CONTROLES.xlsx": ['NOMBRE_CORTO_POZO', 'TEST_DT', 'TEST_PURP_CD', 'BRUTA', 'CHOKE_SIZE','PROD_OIL_24'],
                    "datos_prueba/CNS_NOC_TOW_PAR_PERD.xlsx": ['NOMBRE_CORTO_POZO', 'PROD_DT', 'HORAS_DE_PARO', 'RUBRO', 'GRAN_RUBRO','PERDIDA_PETROLEO'],
                    "datos_prueba/NOC_G&R_PERFIL_UPA_DECLINO.xlsx": ['POZO', 'FECHA', 'BRUTA_(m3/DC)', 'PETRÓLEO_(m3/DC)', 'AGUA_(m3/DC)'], # Added AGUA
                    "datos_prueba/GIDI_POZO.xlsx": ['padre', 'Inicio Fractura', 'SE actual', 'Tipo Aseg IP', 'Zona', 'MPE_rem','Meses_Desplazados'], # Added Meses_Desplazados if it's read directly
                    "datos_prueba/PA_2025_Activos.xlsx": ['Area de Reserva'],
                    "datos_prueba/UPS_FT_PROY_CONSULTA_ACTIVIDAD.xlsx": ['Sigla_Pozo_Cd', 'Fecha_Inicio_Dttm', 'Operacion_Name', 'Area_Reserva_Name'],
                    "datos_prueba/FDD_CNS_NOC_OW_INSTALACIONES.xlsx": ['NOMBRE_POZO', 'COMPONENTE', 'FECHA_INSTALACION'],
                    "datos_prueba/UPS_FT_CABEZA_POZO.xlsx": ['Boca_Pozo_Nombre_Oficial', 'NOMBRE_COMP', 'FECHA_INSTALACION'],
                    "datos_prueba/Diagnostico_FDD_CNS_GRALO_FDP.xlsx": ['NOMBRE_POZO_CORTO', 'FECHA_DIAGNOSTICO', 'DIAGNOSTICO', 'JUSTIFICACION', 'CLASE', 'ZONA'],
                }
            file_paths_ordered = [
                "datos_prueba/UPS_DIM_COMPLETACION.xlsx", "datos_prueba/CNS_NOC_PI.xlsx",
                "datos_prueba/CNS_NOC_TOW_CONTROLES.xlsx", "datos_prueba/CNS_NOC_TOW_PAR_PERD.xlsx",
                "datos_prueba/NOC_G&R_PERFIL_UPA_DECLINO.xlsx", "datos_prueba/GIDI_POZO.xlsx",
                "datos_prueba/PA_2025_Activos.xlsx", "datos_prueba/UPS_FT_PROY_CONSULTA_ACTIVIDAD.xlsx",
                "datos_prueba/FDD_CNS_NOC_OW_INSTALACIONES.xlsx", "datos_prueba/UPS_FT_CABEZA_POZO.xlsx"
            ]

            df_target_names = [ # Nombres de los atributos de la clase donde se guardarán los DFs
                'UPS_DIM_COMPLETACION', 'CNS_NOC_PI', 'CNS_NOC_TOW_CONTROLES', 'CNS_NOC_TOW_PAR_PERD',
                'NOC_GR_PERFIL_UPA_DECLINO', 'GIDI_POZO', 'PA_2025_Activos', 'UPS_FT_PROY_CONSULTA_ACTIVIDAD',
                'FDD_CNS_NOC_OW_INSTALACIONES', 'UPS_FT_CABEZA_POZO'
            ]

            # Crear un mapeo de file_path a nombre de atributo deseado
            file_path_to_attr_name_map = dict(zip(file_paths_ordered, df_target_names))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Guardar el file_path original con cada futuro
                future_to_filepath = {
                    executor.submit(self._read_and_process_excel, fp, columns_to_read_config): fp 
                    for fp in file_paths_ordered
                }

            for future in tqdm(concurrent.futures.as_completed(future_to_filepath), total=len(file_paths_ordered), desc="Procesando archivos Excel"):
                original_file_path = future_to_filepath[future] # Obtener el file_path original
                try:
                    processed_file_path, df = future.result() # _read_and_process_excel ahora devuelve (file_path, df)
                    # Asegurarse de que el processed_file_path coincida con el original_file_path
                    if original_file_path == processed_file_path:
                        attr_name = file_path_to_attr_name_map[original_file_path]
                        setattr(self, attr_name, df)
                    else:
                        # Esto no debería suceder si _read_and_process_excel devuelve el mismo file_path que recibió
                        self.console.print(f"[red]Error de coincidencia de file_path: {original_file_path} vs {processed_file_path}[/red]")
                except Exception as exc:
                    self.console.print(f"[red]Error procesando {original_file_path}: {exc}[/red]")
                    # Asignar un DataFrame vacío al atributo correspondiente para evitar errores posteriores
                    attr_name = file_path_to_attr_name_map.get(original_file_path)
                    if attr_name:
                        setattr(self, attr_name, pd.DataFrame())

            # Additional processing for specific dataframes
            if not self.PA_2025_Activos.empty and 'Area de Reserva' in self.PA_2025_Activos.columns:
                self.PA_2025_Activos['Area de Reserva'] = self.PA_2025_Activos['Area de Reserva'].astype(str).str.upper()
        
            if not self.UPS_FT_PROY_CONSULTA_ACTIVIDAD.empty:
                if 'Operacion_Name' in self.UPS_FT_PROY_CONSULTA_ACTIVIDAD.columns:
                    self.UPS_FT_PROY_CONSULTA_ACTIVIDAD['Operacion_Name'] = self.UPS_FT_PROY_CONSULTA_ACTIVIDAD['Operacion_Name'].astype(str).str.replace('NOC - ', '', regex=False)
                if 'Area_Reserva_Name' in self.UPS_FT_PROY_CONSULTA_ACTIVIDAD.columns:
                    self.UPS_FT_PROY_CONSULTA_ACTIVIDAD['Area_Reserva_Name'] = self.UPS_FT_PROY_CONSULTA_ACTIVIDAD['Area_Reserva_Name'].astype(str).str.replace('LOMA LA LATA NORTE', 'LOMA CAMPANA', regex=False)

            if not self.FDD_CNS_NOC_OW_INSTALACIONES.empty and 'FECHA_INSTALACION' in self.FDD_CNS_NOC_OW_INSTALACIONES.columns:
                self.FDD_CNS_NOC_OW_INSTALACIONES['FECHA_INSTALACION'] = pd.to_datetime(self.FDD_CNS_NOC_OW_INSTALACIONES['FECHA_INSTALACION'], errors='coerce')
                self.FDD_CNS_NOC_OW_INSTALACIONES = self.FDD_CNS_NOC_OW_INSTALACIONES.sort_values('FECHA_INSTALACION', ascending=False).reset_index(drop=True)
                if 'NOMBRE_POZO' in self.FDD_CNS_NOC_OW_INSTALACIONES.columns:
                    self.FDD_CNS_NOC_OW_INSTALACIONES_ultimos = self.FDD_CNS_NOC_OW_INSTALACIONES.sort_values('FECHA_INSTALACION').groupby('NOMBRE_POZO').last().reset_index()

            if not self.UPS_FT_CABEZA_POZO.empty and 'Boca_Pozo_Nombre_Oficial' in self.UPS_FT_CABEZA_POZO.columns:
                self.UPS_FT_CABEZA_POZO['Nombre_Boca_Pozo_Oficial'] = self.UPS_FT_CABEZA_POZO['Boca_Pozo_Nombre_Oficial'].astype(str).str.replace('YPF.Nq.', '', regex=False)
                if 'NOMBRE_COMP' in self.UPS_FT_CABEZA_POZO.columns:
                    self.UPS_FT_CABEZA_POZO_filtrado = self.UPS_FT_CABEZA_POZO[self.UPS_FT_CABEZA_POZO['NOMBRE_COMP'].astype(str).str.contains('ROD LOCK BOP', na=False)]

            if not self.GIDI_POZO.empty and not self.UPS_FT_CABEZA_POZO_filtrado.empty:
                self.GIDI_POZO['Inicio Fractura'] = pd.to_datetime(self.GIDI_POZO['Inicio Fractura'], errors='coerce')
                self.actividad_aseguramiento = self.GIDI_POZO[
                    (self.GIDI_POZO['SE actual'] == 'Bombeo Mecánico') & 
                    (self.GIDI_POZO['Tipo Aseg IP'] == 'AD') & 
                    (~self.GIDI_POZO['padre'].isin(self.UPS_FT_CABEZA_POZO_filtrado['Nombre_Boca_Pozo_Oficial']))
                ][['padre', 'Inicio Fractura', 'Zona']].copy()
                if not self.actividad_aseguramiento.empty:
                    self.actividad_aseguramiento['Fecha Aseg'] = self.actividad_aseguramiento['Inicio Fractura'].apply(lambda x: x - pd.DateOffset(months=1) if pd.notna(x) else pd.NaT)

            if not self.GIDI_POZO.empty:
                def calcular_meses_desplazados_gidi(mpe_rem):
                    if pd.isna(mpe_rem): return None
                    if mpe_rem <= 100: return 1
                    elif 100 < mpe_rem <= 250: return 2
                    elif 250 < mpe_rem <= 400: return 3
                    elif mpe_rem > 400: return 4
                    return None
                self.GIDI_POZO['Meses_Desplazados'] = self.GIDI_POZO['MPE_rem'].apply(calcular_meses_desplazados_gidi)
                self.GIDI_POZO['Inicio Fractura'] = pd.to_datetime(self.GIDI_POZO['Inicio Fractura'], errors='coerce')
                self.GIDI_POZO['fecha_interferencia'] = self.GIDI_POZO['Inicio Fractura'].dt.to_period('M').dt.start_time
            
            if not self.ULTIMA_UPA.empty:
                self.ULTIMA_UPA['FECHA'] = pd.to_datetime(self.ULTIMA_UPA['FECHA'], errors='coerce')
                if 'NOMBRE POZO' in self.ULTIMA_UPA.columns:
                    self.ULTIMA_UPA['NOMBRE POZO'] = self.ULTIMA_UPA['NOMBRE POZO'].astype(str).str.replace('YPF.Nq.', '', regex=False)

            if not self.UPA_actual.empty:
                self.UPA_actual['FECHA'] = pd.to_datetime(self.UPA_actual['FECHA'], errors='coerce')
                if 'NOMBRE POZO' in self.UPA_actual.columns:
                    self.UPA_actual['NOMBRE POZO'] = self.UPA_actual['NOMBRE POZO'].astype(str).str.replace('YPF.Nq.', '', regex=False)

            if not self.UPS_DIM_COMPLETACION.empty:
                self.UPS_DIM_COMPLETACION['Fecha_Inicio_Produccion_Dt'] = pd.to_datetime(self.UPS_DIM_COMPLETACION['Fecha_Inicio_Produccion_Dt'], errors='coerce')
                self.UPS_DIM_COMPLETACION['Campana'] = self.UPS_DIM_COMPLETACION['Fecha_Inicio_Produccion_Dt'].dt.year
                self.UPS_DIM_COMPLETACION = self.UPS_DIM_COMPLETACION[self.UPS_DIM_COMPLETACION['Campana'] > 2016]

            if not self.FDD_CNS_GRALO_FDP_DIAGNOSTICO.empty:
                if 'FECHA_DIAGNOSTICO' in self.FDD_CNS_GRALO_FDP_DIAGNOSTICO.columns:
                    self.FDD_CNS_GRALO_FDP_DIAGNOSTICO['FECHA_DIAGNOSTICO'] = pd.to_datetime(self.FDD_CNS_GRALO_FDP_DIAGNOSTICO['FECHA_DIAGNOSTICO'], errors='coerce')
                if 'NOMBRE_CORTO_POZO' in self.FDD_CNS_GRALO_FDP_DIAGNOSTICO.columns:
                    self.FDD_CNS_GRALO_FDP_DIAGNOSTICO['NOMBRE_CORTO_POZO'] = self.FDD_CNS_GRALO_FDP_DIAGNOSTICO['NOMBRE_CORTO_POZO'].astype(str).str.replace('YPF.Nq.', '', regex=False)
                    self.FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos = self.FDD_CNS_GRALO_FDP_DIAGNOSTICO.sort_values('FECHA_DIAGNOSTICO').groupby('NOMBRE_CORTO_POZO').last().reset_index()
            
            self.console.print("[green]Post-procesamiento completado.[/green]")
            # self.console.print("[green]Selección de columnas completada.[/green]")

        except Exception as e:
            self.console.print(f"[red]Error en la carga y procesamiento de datos Excel: {e}[/red]")


    def run_section_1b_load_from_database(self):
        self.console.rule("[bold blue]1B. Carga y Procesamiento Inicial desde Bases de Datos[/bold blue]")
        
        try: # Inicia el bloque try principal para toda la función
            # Configuración de columnas a leer (reutilizada de la carga de Excel)
            columns_to_read_config = {
                "UPS_DIM_COMPLETACION": ['Completacion_Nombre_Corto', 'Metodo_Produccion_Actual_Cd', 'Bloque_Monitoreo_Nombre', 'Fecha_Inicio_Produccion_Dt'],
                "UPS_FT_DLY_SENSORES_PRESIONES_POZOS": ['Boca_Pozo_Nombre_Oficial', 'Fecha_Hora_Dttm', 'Presion_Cabeza_Pozo_Num', 'Presion_Linea_Produccion_Num'],
                "CNS_NOC_TOW_CONTROLES": ['NOMBRE_CORTO_POZO', 'TEST_DT', 'TEST_PURP_CD', 'BRUTA', 'CHOKE_SIZE','PROD_OIL_24'],
                "CNS_NOC_TOW_PAR_PERD": ['NOMBRE_CORTO_POZO', 'PROD_DT', 'HORAS_DE_PARO', 'RUBRO', 'GRAN_RUBRO','PERDIDA_PETROLEO'],
                "NOC_GR_PERFIL_UPA_DECLINO": ['POZO', 'FECHA', 'BRUTA_(m3/DC)', 'PETRÓLEO_(m3/DC)', 'AGUA_(m3/DC)'],
                "UPS_FT_PROY_CONSULTA_ACTIVIDAD": ['Sigla_Pozo_Cd', 'Fecha_Inicio_Dttm', 'Operacion_Name', 'Area_Reserva_Name'],
                "CNS_NOC_OW_INSTALACIONES": ['NOMBRE_POZO', 'COMPONENTE', 'FECHA_INSTALACION'],
                "UPS_FT_CABEZA_POZO": ['Boca_Pozo_Nombre_Oficial', 'NOMBRE_COMP', 'FECHA_INSTALACION'],
                #"FDD_CNS_GRALO_FDP_DIAGNOSTICO": ['NOMBRE_CORTO_POZO', 'FECHA_DIAGNOSTICO', 'DIAGNOSTICO', 'JUSTIFICACION', 'CLASE', 'ZONA']
            }

            # Mapeo de nombres de DataFrame a nombres de tabla y base de datos
            db_map = {
                'UPS_DIM_COMPLETACION': 'teradata', 
                'UPS_FT_PROY_CONSULTA_ACTIVIDAD': 'teradata',
                'UPS_FT_CABEZA_POZO': 'teradata',
                'UPS_FT_DLY_SENSORES_PRESIONES_POZOS': 'teradata',
                'CNS_NOC_TOW_CONTROLES': 'cns', 'CNS_NOC_TOW_PAR_PERD': 'cns',
#                'NOC_GR_PERFIL_UPA_DECLINO': 'cns', # Asumiendo que NOC es de CNS
                'CNS_NOC_OW_INSTALACIONES': 'cns', # Asumiendo que FDD_CNS es de CNS
                #'FDD_CNS_GRALO_FDP_DIAGNOSTICO': 'cns' # Asumiendo que FDD_CNS es de CNS
            }

            # Generar y ejecutar consultas
            for df_name, db_type in tqdm(db_map.items(), desc="Consultando bases de datos"):
                try:
                    columns_list = columns_to_read_config[df_name.replace('.xlsx', '')]
                    columns_str = ", ".join(f'"{col}"' for col in columns_list) # Usar comillas por si hay mayúsculas/minúsculas
                    
                    # --- LÓGICA DE CONSULTA OPTIMIZADA ---
                    if df_name == 'UPS_FT_DLY_SENSORES_PRESIONES_POZOS' and db_type == 'teradata':
                        # Usar la consulta optimizada específica para esta tabla grande
                        table_name = f"P_DIM_V.{df_name}"
                        query = f"SELECT {columns_str} FROM {table_name} WHERE Servidor_Name = 'NOC' AND Fecha_Hora_Dttm > '2025-06-28 00:00:00'"
                        
                        self.console.print(f"\n[cyan]Ejecutando en Teradata (Consulta Optimizada):[/cyan] {query}")
                        self.console.print("[yellow]Iniciando la descarga de datos desde Teradata... Esto puede tardar varios minutos. Se mostrará el progreso por bloques.[/yellow]")
                        start_time = time.time()
                        
                        # Llamar a la función de conexión con un chunksize para obtener retroalimentación
                        df = conectarse_teradata(query, chunksize=50000) # Pedirá datos en bloques de 50,000 filas
                        
                        end_time = time.time()
                        self.console.print(f"\n[green]Descarga de '{df_name}' completada en {end_time - start_time:.2f} segundos.[/green]")
                    else:
                        # Lógica general para las demás tablas
                        if db_type == 'cns':
                            table_name = f"SAHARA.{df_name}"
                            query = f"SELECT {columns_str} FROM {table_name}"
                            self.console.print(f"\n[cyan]Ejecutando en CNS:[/cyan] {query}")
                            df = conectarse_cns(query)
                        elif db_type == 'teradata':
                            table_name = f"P_DIM_V.{df_name}"
                            query = f"SELECT {columns_str} FROM {table_name}"
                            self.console.print(f"\n[cyan]Ejecutando en Teradata:[/cyan] {query}")
                            df = conectarse_teradata(query)
                    
                    setattr(self, df_name, df)
                    self.console.print(f"[green]OK: DataFrame '{df_name}' cargado desde {db_type.upper()}. Shape: {df.shape}[/green]")

                except Exception as e:
                    self.console.print(f"[red]Error al cargar '{df_name}' desde la base de datos: {e}[/red]")
                    setattr(self, df_name, pd.DataFrame()) # Asignar DF vacío en caso de error

            # Cargar los excels restantes que no se consultan a la BD
            self.console.rule("[bold blue]Cargando archivos Excel restantes[/bold blue]")
            excel_files_to_load = {
                "GIDI_POZO": "datos_prueba/GIDI_POZO.xlsx",

                "NOC_G&R_PERFIL_UPA_DECLINO": "datos_prueba/NOC_G&R_PERFIL_UPA_DECLINO.xlsx", # Added AGUA
                "PA_2025_Activos": "datos_prueba/PA_2025_Activos.xlsx",
                "ULTIMA_UPA": "datos_prueba/UPA_anterior.xlsx",
                "UPA_actual": "datos_prueba/UPA_actual.xlsx",
                "FDD_CNS_GRALO_FDP_DIAGNOSTICO" : "datos_prueba/FDD_CNS_GRALO_FDP_DIAGNOSTICO.xlsx"
            }
 
            for df_name, file_path in tqdm(excel_files_to_load.items(), desc="Cargando Excels restantes"):
                try:
                    df = pd.read_excel(file_path)
                    setattr(self, df_name, df)
                    # self.console.print(f"[green]OK: Archivo Excel '{file_path}' cargado en '{df_name}'. Shape: {df.shape}[/green]") # Comentado
                except Exception as e:
                    self.console.print(f"\n[red]Error cargando el archivo Excel '{file_path}': {e}[/red]") # \n para nueva línea
                    setattr(self, df_name, pd.DataFrame())

            # Aplicar el mismo post-procesamiento que en la sección 1
            self.console.rule("[bold blue]Aplicando post-procesamiento a los datos cargados[/bold blue]")
            # Este bloque es una réplica del post-procesamiento de run_section_1_load_excel_data
            # Se podría refactorizar a una función común si se desea.
            
            # Procesamiento de DataFrames cargados (similar a la sección 1)
            if not self.PA_2025_Activos.empty and 'Area de Reserva' in self.PA_2025_Activos.columns:
                self.PA_2025_Activos['Area de Reserva'] = self.PA_2025_Activos['Area de Reserva'].astype(str).str.upper()
        
            if not self.UPS_FT_PROY_CONSULTA_ACTIVIDAD.empty:
                if 'Operacion_Name' in self.UPS_FT_PROY_CONSULTA_ACTIVIDAD.columns:
                    self.UPS_FT_PROY_CONSULTA_ACTIVIDAD['Operacion_Name'] = self.UPS_FT_PROY_CONSULTA_ACTIVIDAD['Operacion_Name'].astype(str).str.replace('NOC - ', '', regex=False)
                if 'Area_Reserva_Name' in self.UPS_FT_PROY_CONSULTA_ACTIVIDAD.columns:
                    self.UPS_FT_PROY_CONSULTA_ACTIVIDAD['Area_Reserva_Name'] = self.UPS_FT_PROY_CONSULTA_ACTIVIDAD['Area_Reserva_Name'].astype(str).str.replace('LOMA LA LATA NORTE', 'LOMA CAMPANA', regex=False)

            if not self.FDD_CNS_NOC_OW_INSTALACIONES.empty and 'FECHA_INSTALACION' in self.FDD_CNS_NOC_OW_INSTALACIONES.columns:
                self.FDD_CNS_NOC_OW_INSTALACIONES['FECHA_INSTALACION'] = pd.to_datetime(self.FDD_CNS_NOC_OW_INSTALACIONES['FECHA_INSTALACION'], errors='coerce')
                self.FDD_CNS_NOC_OW_INSTALACIONES = self.FDD_CNS_NOC_OW_INSTALACIONES.sort_values('FECHA_INSTALACION', ascending=False).reset_index(drop=True)
                if 'NOMBRE_POZO' in self.FDD_CNS_NOC_OW_INSTALACIONES.columns:
                    self.FDD_CNS_NOC_OW_INSTALACIONES_ultimos = self.FDD_CNS_NOC_OW_INSTALACIONES.sort_values('FECHA_INSTALACION').groupby('NOMBRE_POZO').last().reset_index()

            if not self.UPS_FT_CABEZA_POZO.empty and 'Boca_Pozo_Nombre_Oficial' in self.UPS_FT_CABEZA_POZO.columns:
                self.UPS_FT_CABEZA_POZO['Nombre_Boca_Pozo_Oficial'] = self.UPS_FT_CABEZA_POZO['Boca_Pozo_Nombre_Oficial'].astype(str).str.replace('YPF.Nq.', '', regex=False)
                if 'NOMBRE_COMP' in self.UPS_FT_CABEZA_POZO.columns:
                    self.UPS_FT_CABEZA_POZO_filtrado = self.UPS_FT_CABEZA_POZO[self.UPS_FT_CABEZA_POZO['NOMBRE_COMP'].astype(str).str.contains('ROD LOCK BOP', na=False)]

            if not self.GIDI_POZO.empty and not self.UPS_FT_CABEZA_POZO_filtrado.empty:
                self.GIDI_POZO['Inicio Fractura'] = pd.to_datetime(self.GIDI_POZO['Inicio Fractura'], errors='coerce')
                self.actividad_aseguramiento = self.GIDI_POZO[
                    (self.GIDI_POZO['SE actual'] == 'Bombeo Mecánico') & 
                    (self.GIDI_POZO['Tipo Aseg IP'] == 'AD') & 
                    (~self.GIDI_POZO['padre'].isin(self.UPS_FT_CABEZA_POZO_filtrado['Nombre_Boca_Pozo_Oficial']))
                ][['padre', 'Inicio Fractura', 'Zona']].copy()
                if not self.actividad_aseguramiento.empty:
                    self.actividad_aseguramiento['Fecha Aseg'] = self.actividad_aseguramiento['Inicio Fractura'].apply(lambda x: x - pd.DateOffset(months=1) if pd.notna(x) else pd.NaT)

            if not self.GIDI_POZO.empty:
                def calcular_meses_desplazados_gidi(mpe_rem):
                    if pd.isna(mpe_rem): return None
                    if mpe_rem <= 100: return 1
                    elif 100 < mpe_rem <= 250: return 2
                    elif 250 < mpe_rem <= 400: return 3
                    elif mpe_rem > 400: return 4
                    return None
                self.GIDI_POZO['Meses_Desplazados'] = self.GIDI_POZO['MPE_rem'].apply(calcular_meses_desplazados_gidi)
                self.GIDI_POZO['Inicio Fractura'] = pd.to_datetime(self.GIDI_POZO['Inicio Fractura'], errors='coerce')
                self.GIDI_POZO['fecha_interferencia'] = self.GIDI_POZO['Inicio Fractura'].dt.to_period('M').dt.start_time
            
            if not self.ULTIMA_UPA.empty:
                self.ULTIMA_UPA['FECHA'] = pd.to_datetime(self.ULTIMA_UPA['FECHA'], errors='coerce')
                if 'NOMBRE POZO' in self.ULTIMA_UPA.columns:
                    self.ULTIMA_UPA['NOMBRE POZO'] = self.ULTIMA_UPA['NOMBRE POZO'].astype(str).str.replace('YPF.Nq.', '', regex=False)

            if not self.UPA_actual.empty:
                self.UPA_actual['FECHA'] = pd.to_datetime(self.UPA_actual['FECHA'], errors='coerce')
                if 'NOMBRE POZO' in self.UPA_actual.columns:
                    self.UPA_actual['NOMBRE POZO'] = self.UPA_actual['NOMBRE POZO'].astype(str).str.replace('YPF.Nq.', '', regex=False)

            if not self.UPS_DIM_COMPLETACION.empty:
                self.UPS_DIM_COMPLETACION['Fecha_Inicio_Produccion_Dt'] = pd.to_datetime(self.UPS_DIM_COMPLETACION['Fecha_Inicio_Produccion_Dt'], errors='coerce')
                self.UPS_DIM_COMPLETACION['Campana'] = self.UPS_DIM_COMPLETACION['Fecha_Inicio_Produccion_Dt'].dt.year
                self.UPS_DIM_COMPLETACION = self.UPS_DIM_COMPLETACION[self.UPS_DIM_COMPLETACION['Campana'] > 2016]

            if not self.FDD_CNS_GRALO_FDP_DIAGNOSTICO.empty:
                if 'FECHA_DIAGNOSTICO' in self.FDD_CNS_GRALO_FDP_DIAGNOSTICO.columns:
                    self.FDD_CNS_GRALO_FDP_DIAGNOSTICO['FECHA_DIAGNOSTICO'] = pd.to_datetime(self.FDD_CNS_GRALO_FDP_DIAGNOSTICO['FECHA_DIAGNOSTICO'], errors='coerce')
                if 'NOMBRE_CORTO_POZO' in self.FDD_CNS_GRALO_FDP_DIAGNOSTICO.columns:
                    self.FDD_CNS_GRALO_FDP_DIAGNOSTICO['NOMBRE_CORTO_POZO'] = self.FDD_CNS_GRALO_FDP_DIAGNOSTICO['NOMBRE_CORTO_POZO'].astype(str).str.replace('YPF.Nq.', '', regex=False)
                    self.FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos = self.FDD_CNS_GRALO_FDP_DIAGNOSTICO.sort_values('FECHA_DIAGNOSTICO').groupby('NOMBRE_CORTO_POZO').last().reset_index()
            
            self.console.print("[green]Post-procesamiento completado.[/green]")
            # self.console.print("[green]Selección de columnas completada.[/green]")
        
        except Exception as e: # Captura cualquier error en la función
            self.console.print(f"[bold red]Error en run_section_1b_load_from_database: {e}[/bold red]")


    def _save_df_to_parquet(self, df, file_name_stem, dataframes_to_save_dict): # Pass dict for safety
        try:
            df_to_save = df.copy() # Work on a copy to avoid unintended modifications to the original df in memory

            # Convert object columns to string to prevent Parquet saving issues
            for col_name in df_to_save.select_dtypes(include=['object']).columns:
                df_to_save[col_name] = df_to_save[col_name].astype(str)
            
            import os
            os.makedirs("datos_procesados", exist_ok=True)

            # ---- DEBUGGING: Imprime las columnas y dtypes ANTES de guardar ----
            # Actualizado para FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos
            if file_name_stem == "FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos": 
                self.console.print(f"[cyan]DEBUG: Columns in '{file_name_stem}' (df_to_save) before saving to Parquet: {df_to_save.columns.tolist()}[/cyan]")
                self.console.print(f"[cyan]DEBUG: Info for '{file_name_stem}' (df_to_save) before saving:[/cyan]")
                import io
                buffer = io.StringIO()
                df_to_save.info(buf=buffer, verbose=True) # Use df_to_save
                info_str = buffer.getvalue()
                self.console.print(f"[cyan]{info_str}[/cyan]")
            # ---- FIN DEBUGGING ----
             
            df_to_save.to_parquet(f"datos_procesados/{file_name_stem}.parquet", index=False) # Use df_to_save
        except Exception as e:
            # Use self.console for consistent error reporting
            self.console.print(f"[red]Error saving {file_name_stem} to parquet: {e}[/red]")


    def run_section_2_save_to_parquet(self):
        self.console.rule("[bold blue]2. Guardar DataFrames a Parquet[/bold blue]")
        self.console.print(f"[cyan]DEBUG: Estado de FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos AL INICIO de run_section_2_save_to_parquet: Shape={self.FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos.shape}, Vacío={self.FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos.empty}[/cyan]") # Nuevo DEBUG
        if not hasattr(self, 'UPS_DIM_COMPLETACION') or self.UPS_DIM_COMPLETACION.empty: # Check if data loaded
            self.console.print("[red]No hay datos cargados para guardar. Ejecute la sección 1 primero.[/red]")
            return

        dataframes_to_save_config = {
            "UPS_DIM_COMPLETACION": self.UPS_DIM_COMPLETACION, "CNS_NOC_PI": self.CNS_NOC_PI,
            "CNS_NOC_TOW_CONTROLES": self.CNS_NOC_TOW_CONTROLES, "CNS_NOC_TOW_PAR_PERD": self.CNS_NOC_TOW_PAR_PERD,
            "NOC_GR_PERFIL_UPA_DECLINO": self.NOC_GR_PERFIL_UPA_DECLINO, "GIDI_POZO": self.GIDI_POZO,
            "PA_2025_Activos": self.PA_2025_Activos, "UPS_FT_PROY_CONSULTA_ACTIVIDAD": self.UPS_FT_PROY_CONSULTA_ACTIVIDAD,
            "FDD_CNS_NOC_OW_INSTALACIONES": self.FDD_CNS_NOC_OW_INSTALACIONES, 
            "UPS_FT_CABEZA_POZO_filtrado": self.UPS_FT_CABEZA_POZO_filtrado,
            "actividad_aseguramiento": self.actividad_aseguramiento,
            "ULTIMA_UPA": self.ULTIMA_UPA, "UPA_actual": self.UPA_actual,
            "FDD_CNS_NOC_OW_INSTALACIONES_ultimos": self.FDD_CNS_NOC_OW_INSTALACIONES_ultimos,
            "FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos": self.FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos
        }
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures_map = {
                executor.submit(self._save_df_to_parquet, df, name, dataframes_to_save_config): name 
                for name, df in dataframes_to_save_config.items() if isinstance(df, pd.DataFrame) and not df.empty
            }
            for future in tqdm(concurrent.futures.as_completed(futures_map), total=len(futures_map), desc="Guardando a Parquet"):
                name = futures_map[future]
                try:
                    future.result() # To catch exceptions from the worker
                except Exception as e:
                    self.console.print(f"[red]Error en el thread al guardar {name}: {e}[/red]")
        self.console.print("[green]Sección 2: Guardado a Parquet completado.[/green]")

    def _read_df_from_parquet(self, file_name_stem): # file_name_stem does not include .parquet
        try:
            df = pd.read_parquet(f"datos_procesados/{file_name_stem}.parquet")
            return file_name_stem, df
        except Exception as e:
            # self.console.print(f"[red]Error reading {file_name_stem}.parquet: {e}[/red]")
            return file_name_stem, pd.DataFrame()


    def run_section_3_load_from_parquet(self):
        self.console.rule("[bold blue]3. Cargar DataFrames desde Parquet[/bold blue]")
        
        parquet_file_stems_to_load = [
            "UPS_DIM_COMPLETACION", "CNS_NOC_PI", "CNS_NOC_TOW_CONTROLES", "CNS_NOC_TOW_PAR_PERD",
            "NOC_GR_PERFIL_UPA_DECLINO", "GIDI_POZO", "PA_2025_Activos", "UPS_FT_PROY_CONSULTA_ACTIVIDAD",
            "FDD_CNS_NOC_OW_INSTALACIONES", "UPS_FT_CABEZA_POZO_filtrado", "actividad_aseguramiento",
            "ULTIMA_UPA", "UPA_actual", "FDD_CNS_NOC_OW_INSTALACIONES_ultimos",
            "FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos" # Added this one
        ]

        # loaded_data = {} # This variable is not used
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_name = {executor.submit(self._read_df_from_parquet, name_stem): name_stem for name_stem in parquet_file_stems_to_load}
            for future in tqdm(concurrent.futures.as_completed(future_to_name), total=len(parquet_file_stems_to_load), desc="Cargando desde Parquet"):
                name_stem_res, df_res = future.result()
                
                # Siempre asignar el resultado de _read_df_from_parquet al atributo.
                # df_res contendrá columnas si el archivo Parquet tenía un esquema, incluso si no tenía filas.
                # Si _read_df_from_parquet falló, df_res será un DataFrame vacío sin columnas.
                setattr(self, name_stem_res, df_res)

                # Lógica de advertencia mejorada:
                if df_res.columns.empty and df_res.empty: 
                    # Este caso significa que _read_df_from_parquet probablemente encontró una excepción
                    # y devolvió pd.DataFrame(), o el archivo Parquet en sí no tenía esquema.
                    self.console.print(f"[yellow]Advertencia: No se pudo cargar '{name_stem_res}.parquet' o el archivo está dañado/no es un Parquet válido o no tiene esquema. El DataFrame está vacío y sin columnas.[/yellow]")
                elif df_res.empty: 
                    # Este caso significa que el archivo Parquet se leyó correctamente como una tabla vacía (0 filas, pero con esquema/columnas).
                    self.console.print(f"[yellow]Advertencia: '{name_stem_res}.parquet' se cargó correctamente pero está vacío (0 filas). Columnas cargadas: {df_res.columns.tolist() if not df_res.columns.empty else 'Ninguna'}[/yellow]")
                # else:
                    # Opcional: imprimir éxito para DataFrames no vacíos
                    # self.console.print(f"Cargado {name_stem_res} desde Parquet, shape: {df_res.shape}")

        if hasattr(self, 'UPS_FT_CABEZA_POZO_filtrado') and not self.UPS_FT_CABEZA_POZO_filtrado.empty:
            self.UPS_FT_CABEZA_POZO = self.UPS_FT_CABEZA_POZO_filtrado
        
        self.console.print("[green]Sección 3: Carga desde Parquet completada.[/green]")

    def run_section_show_parquet_df_info(self):
        self.console.rule("[bold blue]Información de DataFrames Cargados desde Parquet[/bold blue]")
        
        # Lista de los nombres de atributos de DataFrame que esperas haber cargado desde Parquet
        # Esta lista debe coincidir con los stems usados en run_section_3_load_from_parquet
        df_attribute_names = [
            "UPS_DIM_COMPLETACION", "CNS_NOC_PI", "CNS_NOC_TOW_CONTROLES", "CNS_NOC_TOW_PAR_PERD",
            "NOC_GR_PERFIL_UPA_DECLINO", "GIDI_POZO", "PA_2025_Activos", "UPS_FT_PROY_CONSULTA_ACTIVIDAD",
            "FDD_CNS_NOC_OW_INSTALACIONES", "UPS_FT_CABEZA_POZO_filtrado", "actividad_aseguramiento",
            "ULTIMA_UPA", "UPA_actual", "FDD_CNS_NOC_OW_INSTALACIONES_ultimos",
            "FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos"
        ]

        for df_name in df_attribute_names:
            if hasattr(self, df_name):
                df = getattr(self, df_name)
                if isinstance(df, pd.DataFrame):
                    self.console.print(f"\n[bold magenta]DataFrame: {df_name}[/bold magenta]")
                    if not df.empty:
                        self.console.print(f"  [cyan]Columnas ({len(df.columns)}):[/cyan] {df.columns.tolist()}")
                        self.console.print(f"  [cyan]Cantidad de datos (filas, columnas):[/cyan] {df.shape}")
                    else:
                        self.console.print(f"  [yellow]El DataFrame está vacío (0 filas).[/yellow]")
                        if not df.columns.empty:
                            self.console.print(f"  [cyan]Columnas ({len(df.columns)}):[/cyan] {df.columns.tolist()}")
                        else:
                            self.console.print(f"  [yellow]El DataFrame no tiene columnas definidas.[/yellow]")
                else:
                    self.console.print(f"\n[yellow]Atributo '{df_name}' no es un DataFrame.[/yellow]")
            else:
                self.console.print(f"\n[red]DataFrame '{df_name}' no encontrado (no es un atributo de la clase).[/red]")
        self.console.print("\n[green]Inspección de DataFrames completada.[/green]")

    def run_section_4_create_pozo_objects(self):
        self.console.rule("[bold blue]4. Creación de Objetos Pozo[/bold blue]")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            required_dfs = ['UPS_DIM_COMPLETACION', 'CNS_NOC_TOW_CONTROLES', 'NOC_GR_PERFIL_UPA_DECLINO', 
                            'UPS_FT_CABEZA_POZO', 'FDD_CNS_NOC_OW_INSTALACIONES', 'GIDI_POZO', 
                            'CNS_NOC_PI', 'CNS_NOC_TOW_PAR_PERD', 'UPS_FT_PROY_CONSULTA_ACTIVIDAD',
                            'ULTIMA_UPA', 'UPA_actual', 'FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos']
        
            for df_name in required_dfs:
                if not hasattr(self, df_name) or getattr(self, df_name).empty:
                    self.console.print(f"[red]Error: DataFrame '{df_name}' no cargado o vacío. Ejecute secciones anteriores.[/red]")
                    return

            self.lista_pozos = []

            # Ensure required columns exist before iterating
            if 'Completacion_Nombre_Corto' not in self.UPS_DIM_COMPLETACION.columns:
                self.console.print("[red]Error: Columna 'Completacion_Nombre_Corto' no encontrada en UPS_DIM_COMPLETACION.[/red]")
                return

            for well_name in tqdm(self.UPS_DIM_COMPLETACION['Completacion_Nombre_Corto'].unique(), desc="Creando objetos Pozo"):
                pozo_data_completacion = self.UPS_DIM_COMPLETACION[self.UPS_DIM_COMPLETACION['Completacion_Nombre_Corto'] == well_name]
                if pozo_data_completacion.empty: continue

                nombre = pozo_data_completacion['Completacion_Nombre_Corto'].values[0]
                sistema_extraccion = pozo_data_completacion['Metodo_Produccion_Actual_Cd'].values[0]
                area = pozo_data_completacion['Bloque_Monitoreo_Nombre'].values[0]
                fecha_PEM_val = pozo_data_completacion['Fecha_Inicio_Produccion_Dt'].values[0]
                fecha_PEM = pd.Timestamp(fecha_PEM_val) if pd.notna(fecha_PEM_val) else pd.NaT

                controles_df = self.CNS_NOC_TOW_CONTROLES[self.CNS_NOC_TOW_CONTROLES['NOMBRE_CORTO_POZO'] == well_name] if 'NOMBRE_CORTO_POZO' in self.CNS_NOC_TOW_CONTROLES else pd.DataFrame()
                declino_df = self.NOC_GR_PERFIL_UPA_DECLINO[self.NOC_GR_PERFIL_UPA_DECLINO['POZO'] == well_name] if 'POZO' in self.NOC_GR_PERFIL_UPA_DECLINO else pd.DataFrame()
            
                # UPS_FT_CABEZA_POZO uses 'Boca_Pozo_Nombre_Oficial' or 'Nombre_Boca_Pozo_Oficial'
                # The script uses 'Nombre_Boca_Pozo_Oficial' after creating it.
                cabeza_de_pozo_df = pd.DataFrame()
                if 'Nombre_Boca_Pozo_Oficial' in self.UPS_FT_CABEZA_POZO.columns:
                    cabeza_de_pozo_df = self.UPS_FT_CABEZA_POZO[self.UPS_FT_CABEZA_POZO['Nombre_Boca_Pozo_Oficial'] == str(well_name)] # No YPF.Nq. prefix needed if already removed
                elif 'Boca_Pozo_Nombre_Oficial' in self.UPS_FT_CABEZA_POZO.columns: # Fallback
                    cabeza_de_pozo_df = self.UPS_FT_CABEZA_POZO[self.UPS_FT_CABEZA_POZO['Boca_Pozo_Nombre_Oficial'] == 'YPF.Nq.' + str(well_name)]


                instalaciones_df = self.FDD_CNS_NOC_OW_INSTALACIONES[self.FDD_CNS_NOC_OW_INSTALACIONES['NOMBRE_POZO'] == well_name] if 'NOMBRE_POZO' in self.FDD_CNS_NOC_OW_INSTALACIONES else pd.DataFrame()
                interferencias_df = self.GIDI_POZO[self.GIDI_POZO['padre'] == well_name] if 'padre' in self.GIDI_POZO else pd.DataFrame()
                perfil_presiones_df = self.CNS_NOC_PI[self.CNS_NOC_PI['NOMBRE_CORTO_POZO'] == well_name] if 'NOMBRE_CORTO_POZO' in self.CNS_NOC_PI else pd.DataFrame()
                downtime_df = self.CNS_NOC_TOW_PAR_PERD[self.CNS_NOC_TOW_PAR_PERD['NOMBRE_CORTO_POZO'] == well_name] if 'NOMBRE_CORTO_POZO' in self.CNS_NOC_TOW_PAR_PERD else pd.DataFrame()
            
                fecha_crono_series = self.UPS_FT_PROY_CONSULTA_ACTIVIDAD.loc[self.UPS_FT_PROY_CONSULTA_ACTIVIDAD['Sigla_Pozo_Cd'] == well_name, 'Fecha_Inicio_Dttm'] if 'Sigla_Pozo_Cd' in self.UPS_FT_PROY_CONSULTA_ACTIVIDAD else pd.Series(dtype='datetime64[ns]')
            
                fecha_ultima_UPA_series = self.ULTIMA_UPA.loc[self.ULTIMA_UPA['NOMBRE POZO'] == well_name, 'FECHA'] if 'NOMBRE POZO' in self.ULTIMA_UPA else pd.Series(dtype='datetime64[ns]')
                fecha_UPA_actual_series = self.UPA_actual.loc[self.UPA_actual['NOMBRE POZO'] == well_name, 'FECHA'] if 'NOMBRE POZO' in self.UPA_actual else pd.Series(dtype='datetime64[ns]')
            
                diagnostico_df_filtered = self.FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos[self.FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos['NOMBRE_CORTO_POZO'] == well_name] if 'NOMBRE_CORTO_POZO' in self.FDD_CNS_GRALO_FDP_DIAGNOSTICO_ultimos else pd.DataFrame()

                pozo_obj = pozo(nombre, sistema_extraccion, controles_df, declino_df, cabeza_de_pozo_df, 
                                instalaciones_df, fecha_PEM, interferencias_df, perfil_presiones_df, downtime_df, area,
                                fecha_ultima_UPA_series, fecha_UPA_actual_series, diagnostico_df_filtered,
                                self.UPA_actual, self.ULTIMA_UPA) # Pass full UPA_actual for 'Días de Cierre'
            
                pozo_obj.calcular_fecha_capex()
                pozo_obj.set_fecha_crono(pd.Timestamp(fecha_crono_series.iloc[0]) if not fecha_crono_series.empty else pd.NaT)
            
                self.lista_pozos.append(pozo_obj)
        
            self.console.print(f"[green]Sección 4: Creación de {len(self.lista_pozos)} objetos Pozo completada.[/green]")

    def run_section_inspect_pozo(self):
        self.console.rule("[bold blue]Inspeccionar Objeto Pozo[/bold blue]")
        if not hasattr(self, 'lista_pozos') or not self.lista_pozos:
            self.console.print("[red]La lista de pozos no ha sido creada. Ejecute la Sección 4 primero.[/red]")
            return

        well_name_input = Prompt.ask("Ingrese el nombre del pozo a inspeccionar")
        normalized_input = self._normalize_well_name(well_name_input)
        
        found_pozo_obj = None
        for pozo_obj in self.lista_pozos:
            normalized_pozo_attr_name = self._normalize_well_name(pozo_obj.nombre)
            if normalized_pozo_attr_name == normalized_input:
                found_pozo_obj = pozo_obj
                break
        
        if found_pozo_obj:
            self.console.print(f"\n[bold green]Mostrando atributos para el pozo: {found_pozo_obj.nombre}[/bold green]")
            
            table = Table(title=f"Atributos del Pozo: {found_pozo_obj.nombre}", show_lines=True)
            table.add_column("Atributo", style="cyan", overflow="fold")
            table.add_column("Valor", style="magenta", overflow="fold")

            attributes = vars(found_pozo_obj)
            dataframes_to_print_details = {} # Para almacenar DataFrames y sus nombres

            for attr_name, attr_value in attributes.items():
                val_repr = ""
                if isinstance(attr_value, pd.DataFrame):
                    if not attr_value.empty:
                        cols_preview = attr_value.columns.tolist()
                        cols_display = f"{cols_preview[:5]}..." if len(cols_preview) >  5 else cols_preview
                        val_repr = f"DataFrame (Shape: {attr_value.shape}, Columnas: {cols_display}) - Ver detalles abajo"
                        dataframes_to_print_details[attr_name] = attr_value # Guardar para imprimir después
                    else:
                        val_repr = f"DataFrame (Vacío, Columnas: {attr_value.columns.tolist()})"
                elif isinstance(attr_value, pd.Series):
                    if not attr_value.empty:
                        val_repr = f"Serie (Longitud: {len(attr_value)}, Primeros valores: {attr_value.head(3).to_dict() if len(attr_value) > 0 else 'N/A'})"
                    else:
                        val_repr = "Serie (Vacía)"
                elif isinstance(attr_value, list) or isinstance(attr_value, dict):
                    str_val = str(attr_value)
                    if len(str_val) > 150:
                         val_repr = str_val[:150] + "..."
                    else:
                        val_repr = str_val
                else:
                    val_repr = str(attr_value)
                    if len(val_repr) > 150: # Truncar valores de string largos para la tabla
                        val_repr = val_repr[:150] + "..."
                
                table.add_row(attr_name, val_repr)
            
            self.console.print(table)

            if dataframes_to_print_details:
                self.console.print("\n[bold yellow]Detalles de los DataFrames asociados:[/bold yellow]")
                for df_attr_name, df_value in dataframes_to_print_details.items():
                    self.console.print(f"\n[bold cyan]DataFrame: {df_attr_name}[/bold cyan] (Primeras 5 filas)")
                    # Para evitar que la tabla de Rich intente formatear el DataFrame de pandas de forma extraña,
                    # lo convertimos a string antes de imprimirlo con self.console.print.
                    # O, mejor aún, usamos la propia capacidad de Rich para imprimir DataFrames si es simple.
                    # Si el DataFrame es muy ancho, puede que no se vea bien.
                    # Considerar usar df_value.to_string() para una representación más controlada si es necesario.
                    with self.console.capture() as capture: # Captura la salida de print(df) para que Rich la maneje
                        print(df_value.head())
                    self.console.print(capture.get())
                    # También podrías imprimir df_value.info()
                    # self.console.print(f"[bold cyan]Info para {df_attr_name}:[/bold cyan]")
                    # buffer = io.StringIO()
                    # df_value.info(buf=buffer)
                    # self.console.print(buffer.getvalue())


        else:
            self.console.print(f"[yellow]Pozo '{well_name_input}' (normalizado a '{normalized_input}') no encontrado en la lista_pozos.[/yellow]")


    def run_section_5_create_upa_plans(self):
        self.console.rule("[bold blue]5. Creación de Planes UPA[/bold blue]")
        if not self.lista_pozos:
            self.console.print("[red]Lista de pozos no creada. Ejecute la sección 4 primero.[/red]")
            return

        pozos_cbm = [p for p in self.lista_pozos if p.actividad == 'CBM']
        pozos_bif = [p for p in self.lista_pozos if p.actividad == 'BIF']

        # Filter by campana (year of fecha_PEM)
        # Ensure pozo.campana is an integer year
        relevant_years = [2016,2017,2018,2019,2020,2021,2022,2023, 2024, 2025]
        pozos_cbm_filtrados = [p for p in pozos_cbm if p.campana in relevant_years]
        pozos_bif_filtrados = [p for p in pozos_bif if p.campana in relevant_years]

        # Sort
        pozos_cbm_ordenados = sorted(pozos_cbm_filtrados, key=lambda x: (x.ultimo_control or float('-inf'), x.presion_cabeza_actual or float('-inf'), x.orificio_ult_control or float('-inf'), x.relacion_presion_linea or float('-inf')))
        pozos_bif_ordenados = sorted(pozos_bif_filtrados, key=lambda x: (x.ultimo_control or float('-inf'), x.presion_cabeza_actual or float('-inf'), x.orificio_ult_control or float('-inf'), x.relacion_presion_linea or float('-inf')))

        # Filter out pozos without fecha_capex
        pozos_bif_ordenados = [p for p in pozos_bif_ordenados if pd.notna(p.fecha_capex)]
        pozos_cbm_ordenados = [p for p in pozos_cbm_ordenados if pd.notna(p.fecha_capex)]

        # Display counts (optional)
        # bloques_bif = Counter([p.area for p in pozos_bif_ordenados])
        # ... (display logic from notebook) ...

        # Create df_pozos for export
        df_pozos_data = []
        for p_list in [pozos_bif_ordenados, pozos_cbm_ordenados]:
            for p in p_list:
                df_pozos_data.append({
                    'Nombre': p.nombre, 'Presion de Cabeza': p.presion_cabeza_actual, 
                    'Relacion de Presion de Linea': p.relacion_presion_linea, 'Fecha Capex': p.fecha_capex, 
                    'Fecha Crono': p.fecha_crono, 'Fecha Capex Ajustada': p.fecha_capex_ajustada, 
                    'Estado de Actividad': p.estado_actividad, 'Estado Operativo': p.estado_operativo, 
                    'Ultimo Control': p.ultimo_control, 'Orificio Ultimo Control': p.orificio_ult_control, 
                    'Fecha Ultimo Control': p.fecha_ultimo_control, 'Area': p.area, 
                    'Actividad': p.actividad, 'Campana': p.campana, 'Fecha ultima UPA': p.fecha_upa_anterior
                })
        df_pozos_export = pd.DataFrame(df_pozos_data)
        try:
            import os
            os.makedirs("Resultados", exist_ok=True)
            df_pozos_export.to_excel('Resultados/Pozos_BIF_CBM_campanas_filtradas.xlsx', index=False)
            self.console.print("[green]Exportado 'Pozos_BIF_CBM_campanas_filtradas.xlsx'[/green]")
        except Exception as e:
            self.console.print(f"[red]Error exportando df_pozos_export: {e}[/red]")


        # UPA Plan Logic
        max_bif_mensuales = [0, 0, 0, 0, 0, 19, 19, 19, 19, 19, 19, 19]
        max_cbm_mensuales = [0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]
        fecha_inicial_plan = pd.Timestamp('2025-01-01')
        fecha_final_plan = pd.Timestamp('2025-12-01') # Adjusted to match original logic (12 months for 2025)
        rango_fechas_plan = pd.date_range(start=fecha_inicial_plan, end=fecha_final_plan, freq='MS')

        # UPA sin limite
        upa_sin_limite_list = []
        for fecha_mes in rango_fechas_plan:
            mes_period = fecha_mes.to_period('M')
            pozos_bif_mes_sl = [p for p in pozos_bif_ordenados if pd.notna(p.fecha_capex_ajustada) and pd.Timestamp(p.fecha_capex_ajustada).to_period('M') == mes_period]
            pozos_cbm_mes_sl = [p for p in pozos_cbm_ordenados if pd.notna(p.fecha_capex_ajustada) and pd.Timestamp(p.fecha_capex_ajustada).to_period('M') == mes_period]
            upa_sin_limite_list.append({'Mes': mes_period, 'Pozos_BIF': len(pozos_bif_mes_sl), 'Pozos_CBM': len(pozos_cbm_mes_sl)})
        
        self.df_upa_sin_limite = pd.DataFrame(upa_sin_limite_list)
        
        data_upa_sin_limite_export = []
        # Iterating over original sorted lists to build the export based on their fecha_capex_ajustada
        all_pozos_ordenados_sl = pozos_bif_ordenados + pozos_cbm_ordenados
        for p in all_pozos_ordenados_sl:
            if pd.notna(p.fecha_capex_ajustada) and fecha_inicial_plan <= pd.Timestamp(p.fecha_capex_ajustada) <= fecha_final_plan: # Ensure it's within plan range
                 data_upa_sin_limite_export.append([
                    p.nombre, p.actividad, p.fecha_capex_ajustada, p.meses_desplazados, p.area, 
                    p.presion_cabeza_actual, p.ultimo_control, p.fecha_ultimo_control, p.estado_actividad, 
                    p.estado_operativo, p.ultimo_paro_rubro, p.ultimo_paro_fecha, p.tiempo_ultimo_paro,
                    p.fecha_capex, p.bruta_declino_inicial, p.correccion, p.fecha_upa_anterior # bruta_declino_inicial was 0 in original
                ])
        # CORRECCIÓN: Se arregló un error de tipeo en la lista de columnas. Había una coma dentro de 'Tiempo Ultimo Paro,'
        # que causaba que se uniera con la siguiente columna, resultando en 16 nombres de columna para 17 columnas de datos.
        self.df_upa_sin_limite_export = pd.DataFrame(data_upa_sin_limite_export, columns=['Completacion', 'Actividad', 'Fecha Capex Ajustada', 'Meses Desplazados', 'Bloque_Monitoreo_Nombre', 'PRESION_CABEZA', 'BRUTA', 'Fecha_BRUTA', 'Estado Actividad', 'Estado Operativo', 'Ultimo Paro Rubro', 'Ultimo Paro Fecha', 'Tiempo Ultimo Paro',
        'Fecha Capex Original','BRUTA_DECLINO_INICIAL','Correccion','Fecha UPA Anterior'])
        self.df_upa_sin_limite_export = self.df_upa_sin_limite_export.reset_index(drop=True)
        self.df_upa_sin_limite_export['Index'] = self.df_upa_sin_limite_export.index + 1

        for p in self.lista_pozos: # Assign priority to all pozos in self.lista_pozos
            matching_rows = self.df_upa_sin_limite_export[self.df_upa_sin_limite_export['Completacion'] == p.nombre]
            if not matching_rows.empty:
                p.set_prioridad(matching_rows['Index'].iloc[0])


        # UPA con limite
        # Create copies of sorted lists for modification during limited plan
        pozos_bif_plan_limitado = [p for p in pozos_bif_ordenados if pd.notna(p.fecha_capex_ajustada)]
        pozos_cbm_plan_limitado = [p for p in pozos_cbm_ordenados if pd.notna(p.fecha_capex_ajustada)]

        upa_con_limite_list = []
        data_upa_con_limite_export = []

        for fecha_mes_cl in rango_fechas_plan:
            mes_period_cl = fecha_mes_cl.to_period('M')
            
            # Pozos scheduled for this month (before limits)
            current_bif_for_month = sorted([p for p in pozos_bif_plan_limitado if pd.Timestamp(p.fecha_capex_ajustada).to_period('M') == mes_period_cl], key=lambda x: x.prioridad if pd.notna(x.prioridad) else float('inf'))
            current_cbm_for_month = sorted([p for p in pozos_cbm_plan_limitado if pd.Timestamp(p.fecha_capex_ajustada).to_period('M') == mes_period_cl], key=lambda x: x.prioridad if pd.notna(x.prioridad) else float('inf'))

            # Apply monthly limits
            bif_this_month_actual = current_bif_for_month[:max_bif_mensuales[mes_period_cl.month - 1]]
            cbm_this_month_actual = current_cbm_for_month[:max_cbm_mensuales[mes_period_cl.month - 1]]
            
            upa_con_limite_list.append({'Mes': mes_period_cl, 'Pozos_BIF': len(bif_this_month_actual), 'Pozos_CBM': len(cbm_this_month_actual)})

            # Add selected pozos to export list
            for p_list_actual in [bif_this_month_actual, cbm_this_month_actual]:
                for p_actual in p_list_actual:
                    declino_val = p_actual.search_declino(p_actual.fecha_capex_ajustada)
                    data_upa_con_limite_export.append([
                        p_actual.nombre, p_actual.actividad, p_actual.fecha_capex_ajustada, p_actual.meses_desplazados, p_actual.area, 
                        p_actual.presion_cabeza_actual, p_actual.ultimo_control, p_actual.fecha_ultimo_control, p_actual.estado_actividad, 
                        p_actual.estado_operativo, p_actual.ultimo_paro_rubro, p_actual.ultimo_paro_fecha, p_actual.tiempo_ultimo_paro,
                        p_actual.fecha_capex, p_actual.bruta_declino_inicial, p_actual.correccion, p_actual.fecha_upa_anterior, p_actual.prioridad, declino_val
                    ])
            
            # Pozos to be deferred
            bif_deferred = current_bif_for_month[max_bif_mensuales[mes_period_cl.month - 1]:]
            cbm_deferred = current_cbm_for_month[max_cbm_mensuales[mes_period_cl.month - 1]:]

            next_month_start = (mes_period_cl + 1).start_time
            for p_def_list in [bif_deferred, cbm_deferred]:
                for p_def in p_def_list:
                    p_def.fecha_capex_ajustada = next_month_start
                    p_def.verificar_interferencias(p_def.fecha_capex_ajustada) # Re-check after deferral

        # Final verification pass for all pozos after deferrals
        all_pozos_for_final_verification = pozos_bif_plan_limitado + pozos_cbm_plan_limitado
        for p_final_verif in all_pozos_for_final_verification:
             if pd.notna(p_final_verif.fecha_capex_ajustada):
                p_final_verif.verificar_interferencias(p_final_verif.fecha_capex_ajustada)


        self.df_upa_con_limite = pd.DataFrame(data_upa_con_limite_export, columns=['Completacion', 'Actividad', 'Fecha Capex Ajustada', 'Meses Desplazados', 'Bloque_Monitoreo_Nombre', 'PRESION_CABEZA', 'BRUTA', 'Fecha_BRUTA', 'Estado Actividad', 'Estado Operativo', 'Ultimo Paro Rubro', 'Ultimo Paro Fecha', 'Tiempo Ultimo Paro', 'Fecha Capex Original', 'BRUTA_DECLINO_INICIAL', 'Correccion','Fecha UPA Anterior','Index','Fecha_declino_oil'])

        try:
            self.df_upa_sin_limite.to_excel('Resultados/UPA_sin_limite_recurso.xlsx', index=False) # df_upa_sin_limite is the summary table
            self.df_upa_sin_limite_export.to_excel('Resultados/UPA_sin_limite_detalle_pozos.xlsx', index=False) # Detail table
            self.console.print("[green]Exportado 'UPA_sin_limite_recurso.xlsx' y detalle.[/green]")
            
            # self.df_upa_con_limite is the detailed list, UPA_con_limite_list is the summary
            pd.DataFrame(upa_con_limite_list).to_excel('Resultados/UPA_limitada_recursos_resumen.xlsx', index=False)
            self.df_upa_con_limite.to_excel('Resultados/UPA_limitada_recursos_detalle_pozos.xlsx', index=False)
            self.console.print("[green]Exportado 'UPA_limitada_recursos_resumen.xlsx' y detalle.[/green]")
        except Exception as e:
            self.console.print(f"[red]Error exportando planes UPA: {e}[/red]")

        self.console.print("[green]Sección 5: Creación de Planes UPA completada.[/green]")


    def run_section_6_process_aseguramiento(self):
        self.console.rule("[bold blue]6. Procesamiento de Aseguramiento de Pozos[/bold blue]")
        if not self.lista_pozos:
            self.console.print("[red]Lista de pozos no creada. Ejecute la sección 4 primero.[/red]")
            return

        pozos_con_interferencias_ad_sr = []
        for p in self.lista_pozos:
            if p.interferencias is not None and not p.interferencias.empty and \
               'Tipo Aseg IP' in p.interferencias.columns and \
               p.interferencias['Tipo Aseg IP'].astype(str).str.contains('AD', case=False, na=False).any() and \
               p.sistema_extraccion == 'SR':
                pozos_con_interferencias_ad_sr.append(p)
        
        data_pozos_con_interferencias_export = []
        for p in pozos_con_interferencias_ad_sr:
            # Assuming 'PAD hijo' and 'Inicio Fractura' exist and are single values for these relevant interferences
            # This might need refinement if multiple 'AD' interferences exist for a pozo
            relevant_interference = p.interferencias[p.interferencias['Tipo Aseg IP'].astype(str).str.contains('AD', case=False, na=False)].iloc[0] if not p.interferencias[p.interferencias['Tipo Aseg IP'].astype(str).str.contains('AD', case=False, na=False)].empty else pd.Series()
            
            pad_hijo = relevant_interference.get('PAD hijo', None) # Use .get for safety
            inicio_fractura = pd.to_datetime(relevant_interference.get('Inicio Fractura', pd.NaT), errors='coerce')
            fecha_aseguramiento = (inicio_fractura.to_period('M').start_time - pd.DateOffset(months=1)) if pd.notna(inicio_fractura) else pd.NaT

            data_pozos_con_interferencias_export.append([
                p.nombre, p.sistema_extraccion, pad_hijo, fecha_aseguramiento, p.rodlock, p.fecha_rodlock
            ])
        
        df_pozos_con_interferencias_export = pd.DataFrame(data_pozos_con_interferencias_export, columns=['Nombre', 'Sistema Extraccion', 'PAD Hijo', 'Fecha Aseguramiento','Rodlock','Fecha Rodlock'])

        pozos_sin_rodlock_list = [p for p in pozos_con_interferencias_ad_sr if p.rodlock == 0 or pd.isna(p.rodlock)]
        pozos_con_rodlock_list = [p for p in pozos_con_interferencias_ad_sr if p.rodlock is not None and p.rodlock > 0]

        # Data for pozos sin rodlock
        data_pozos_sin_rodlock_export = []
        for p in pozos_sin_rodlock_list:
            relevant_interference = p.interferencias[p.interferencias['Tipo Aseg IP'].astype(str).str.contains('AD', case=False, na=False)].iloc[0] if not p.interferencias[p.interferencias['Tipo Aseg IP'].astype(str).str.contains('AD', case=False, na=False)].empty else pd.Series()
            pad_hijo = relevant_interference.get('PAD hijo', None)
            inicio_fractura = pd.to_datetime(relevant_interference.get('Inicio Fractura', pd.NaT), errors='coerce')
            fecha_aseguramiento = (inicio_fractura.to_period('M').start_time - pd.DateOffset(months=1)) if pd.notna(inicio_fractura) else pd.NaT
            data_pozos_sin_rodlock_export.append([p.nombre, p.sistema_extraccion, pad_hijo, fecha_aseguramiento, p.asegurado_armadura])
        df_pozos_sin_rodlock_export = pd.DataFrame(data_pozos_sin_rodlock_export, columns=['Nombre', 'Sistema Extraccion', 'PAD Hijo', 'Fecha Aseguramiento','Asegurado con armadura'])

        # Data for pozos con rodlock
        data_pozos_con_rodlock_export = []
        for p in pozos_con_rodlock_list:
            relevant_interference = p.interferencias[p.interferencias['Tipo Aseg IP'].astype(str).str.contains('AD', case=False, na=False)].iloc[0] if not p.interferencias[p.interferencias['Tipo Aseg IP'].astype(str).str.contains('AD', case=False, na=False)].empty else pd.Series()
            pad_hijo = relevant_interference.get('PAD hijo', None)
            inicio_fractura = pd.to_datetime(relevant_interference.get('Inicio Fractura', pd.NaT), errors='coerce')
            fecha_aseguramiento = (inicio_fractura.to_period('M').start_time - pd.DateOffset(months=1)) if pd.notna(inicio_fractura) else pd.NaT
            data_pozos_con_rodlock_export.append([p.nombre, p.sistema_extraccion, pad_hijo, fecha_aseguramiento, p.rodlock, p.fecha_rodlock])
        df_pozos_con_rodlock_export = pd.DataFrame(data_pozos_con_rodlock_export, columns=['Nombre', 'Sistema Extraccion', 'PAD Hijo', 'Fecha Aseguramiento','Rodlock','Fecha Rodlock'])

        try:
            import os
            os.makedirs("Resultados", exist_ok=True)
            df_pozos_con_interferencias_export.to_excel('Resultados/Pozos_con_interferencias_AD_SR.xlsx', index=False)
            df_pozos_sin_rodlock_export.to_excel('Resultados/Pozos_sin_rodlock_AD_SR.xlsx', index=False)
            df_pozos_con_rodlock_export.to_excel('Resultados/Pozos_con_rodlock_AD_SR.xlsx', index=False)
            self.console.print("[green]Exportados archivos de aseguramiento.[/green]")
        except Exception as e:
            self.console.print(f"[red]Error exportando archivos de aseguramiento: {e}[/red]")
        
        self.console.print("[green]Sección 6: Procesamiento de Aseguramiento completado.[/green]")


    def run_section_7_calculate_edt_losses(self):
        self.console.rule("[bold blue]7. Cálculo de Pérdidas EDT[/bold blue]")
        if not hasattr(self, 'CNS_NOC_TOW_PAR_PERD') or self.CNS_NOC_TOW_PAR_PERD.empty or \
           not hasattr(self, 'CNS_NOC_TOW_CONTROLES') or self.CNS_NOC_TOW_CONTROLES.empty:
            self.console.print("[red]DataFrames CNS_NOC_TOW_PAR_PERD o CNS_NOC_TOW_CONTROLES no cargados. Ejecute secciones anteriores.[/red]")
            return

        cns_par_perd = self.CNS_NOC_TOW_PAR_PERD.copy()
        cns_controles = self.CNS_NOC_TOW_CONTROLES.copy()

        if 'PROD_DT' not in cns_par_perd.columns or 'HORAS_DE_PARO' not in cns_par_perd.columns:
            self.console.print("[red]Columnas requeridas faltantes en CNS_NOC_TOW_PAR_PERD.[/red]")
            return
        
        cns_par_perd['PROD_DT'] = pd.to_datetime(cns_par_perd['PROD_DT'], errors='coerce')
        ultima_fecha_par_perd = cns_par_perd['PROD_DT'].max()
        
        pozos_parados = cns_par_perd[
            (cns_par_perd['PROD_DT'] == ultima_fecha_par_perd) & 
            (cns_par_perd['HORAS_DE_PARO'] == 24)
        ]

        if 'RUBRO' not in pozos_parados.columns:
            self.console.print("[red]Columna 'RUBRO' faltante en pozos_parados.[/red]")
            return

        rubro_filtro = pozos_parados['RUBRO'].astype(str).str.contains('Espera Tractor|Estudio extracción|Tractor', regex=True, na=False)
        pozos_filtrados_edt = pozos_parados[rubro_filtro]

        if 'NOMBRE_CORTO_POZO' not in cns_controles.columns or 'TEST_DT' not in cns_controles.columns:
            self.console.print("[red]Columnas requeridas faltantes en CNS_NOC_TOW_CONTROLES.[/red]")
            return
        
        cns_controles['TEST_DT'] = pd.to_datetime(cns_controles['TEST_DT'], errors='coerce')
        ultimo_control_df = cns_controles.loc[cns_controles.groupby('NOMBRE_CORTO_POZO')['TEST_DT'].idxmax()]

        pozos_filtrados_edt = pozos_filtrados_edt.merge(
            ultimo_control_df[['NOMBRE_CORTO_POZO', 'PROD_OIL_24']], 
            on='NOMBRE_CORTO_POZO', 
            how='left'
        )
        
        # Ensure PERDIDA_PETROLEO is numeric for sorting
        pozos_filtrados_edt['PERDIDA_PETROLEO'] = pd.to_numeric(pozos_filtrados_edt.get('PERDIDA_PETROLEO'), errors='coerce').fillna(0)
        pozos_filtrados_edt['PROD_OIL_24'] = pd.to_numeric(pozos_filtrados_edt.get('PROD_OIL_24'), errors='coerce').fillna(0)


        pozos_priorizados_edt = pozos_filtrados_edt.sort_values(by=['PERDIDA_PETROLEO', 'PROD_OIL_24'], ascending=[False, False])
        
        pozos_priorizados_edt['fecha_PU'] = pd.NaT
        pozos_priorizados_edt['fecha_EDT'] = pd.NaT

        fecha_inicio_edt = pd.to_datetime('2025-06-01')
        fechas_edt_calc = pd.date_range(start=fecha_inicio_edt, end='2025-12-01', freq='MS') # Corrected end date
        
        resultados_edt = []
        for fecha_calc in fechas_edt_calc:
            dias_mes_calc = fecha_calc.days_in_month
            temp_df = pozos_priorizados_edt.copy() # Work on a copy for each month's calculation
            temp_df['P_EDT'] = temp_df['PERDIDA_PETROLEO'] * dias_mes_calc
            temp_df['fecha_EDT'] = fecha_calc # Assign the first day of the month
            resultados_edt.append(temp_df)

        plan_perdidas_EDT_conocidas = pd.concat(resultados_edt, ignore_index=True)
        
        # The logic for df_operativos_balanceados and filtering by fecha_PU was commented out
        # in the original notebook, so it's omitted here unless specified.

        try:
            import os
            os.makedirs("Resultados", exist_ok=True)
            plan_perdidas_EDT_conocidas.to_excel('Resultados/Perdidas_EDT_conocidas.xlsx', index=False)
            self.console.print("[green]Exportado 'Perdidas_EDT_conocidas.xlsx'[/green]")

            apertura_plan_perdidas_EDT_pivot = plan_perdidas_EDT_conocidas.pivot_table(
                index='fecha_EDT', columns='NOMBRE_CORTO_POZO', values='P_EDT', aggfunc='first' # Use first if unique per pozo-fecha
            )
            apertura_plan_perdidas_EDT_pivot.to_excel("Resultados/Tabla_pivot_EDT_Conocidas.xlsx")
            self.console.print("[green]Exportado 'Tabla_pivot_EDT_Conocidas.xlsx'[/green]")

        except Exception as e:
            self.console.print(f"[red]Error exportando resultados EDT: {e}[/red]")

        # Further EDT optimization logic (IFD, parque, etc.)
        IFD_config = {"LC": 0.3, "LACh": 0.3, "BS": 0.43}
        parque_config = {"LC": 235, "LACh": 73, "BS": 24}
        caudal_fallas_config = {"LC": 16, "LACh": 20, "BS": 22}

        cantidad_fallas_mensuales_calc = {
            activo: round(float((parque_config[activo] * IFD_config[activo]) / 12), 0)
            for activo in parque_config.keys()
        }

        apertura_plan_perdidas_edt_opt = plan_perdidas_EDT_conocidas[['NOMBRE_CORTO_POZO', 'fecha_EDT', 'P_EDT']].copy()
        
        rango_fechas_opt_edt = pd.date_range(start=fecha_inicio_edt, end='2025-12-01', freq='MS') # Consistent range

        for activo, fallas_mensuales_val in cantidad_fallas_mensuales_calc.items():
            for i in range(1, int(fallas_mensuales_val) + 1): # Iterate for each generic well
                for mes_opt in rango_fechas_opt_edt:
                    new_row = pd.DataFrame([{
                        'NOMBRE_CORTO_POZO': f'GENERIC_{activo}_{i}', # Unique generic name
                        'fecha_EDT': mes_opt,
                        'P_EDT': caudal_fallas_config[activo] * mes_opt.days_in_month # P_EDT for full month
                    }])
                    apertura_plan_perdidas_edt_opt = pd.concat([apertura_plan_perdidas_edt_opt, new_row], ignore_index=True)
        
        # Ensure 'Cantidad_Fallas' matches the length of your planning horizon (rango_fechas_opt_edt)
        # Original: [7, 5, 5, 11, 18, 18, 20, 23, 21] - this is 9 months.
        # If rango_fechas_opt_edt is shorter (e.g. June-Dec = 7 months), adjust.
        cantidad_fallas_max_mensual_list = [7, 5, 5, 11, 18, 18, 20, 23, 21] 
        
        # Adjust to the length of rango_fechas_opt_edt
        num_months_in_plan = len(rango_fechas_opt_edt)
        if num_months_in_plan < len(cantidad_fallas_max_mensual_list):
            cantidad_fallas_max_mensual_series = pd.Series(cantidad_fallas_max_mensual_list[:num_months_in_plan])
        else: # Repeat or extend if plan is longer
            cantidad_fallas_max_mensual_series = pd.Series(
                (cantidad_fallas_max_mensual_list * (num_months_in_plan // len(cantidad_fallas_max_mensual_list) + 1))[:num_months_in_plan]
            )

        cantidad_fallas_maximas_mensual_df_edt = pd.DataFrame({
            "Mes": rango_fechas_opt_edt,
            "Cantidad_Fallas": cantidad_fallas_max_mensual_series.values
        })

        apertura_plan_perdidas_EDT_pivot_plan_opt = apertura_plan_perdidas_edt_opt.pivot_table(
            index='fecha_EDT', columns='NOMBRE_CORTO_POZO', values='P_EDT', aggfunc='sum'
        ).fillna(0) # Fill NaNs with 0 before bfill/ffill

        # Fill NaNs carefully - only if a pozo has an entry later and should be filled back
        # For now, let's assume 0 means no production/loss for that pozo in that month
        # The bfill/ffill might not be appropriate if a pozo truly starts/stops.
        # Original code used it, so replicating:
        apertura_plan_perdidas_EDT_pivot_plan_opt = apertura_plan_perdidas_EDT_pivot_plan_opt.fillna(method='bfill').fillna(method='ffill').fillna(0)


        apertura_plan_perdidas_edt_opt = apertura_plan_perdidas_EDT_pivot_plan_opt.copy()
        # pozos_eliminados_df_edt = pd.DataFrame(index=apertura_plan_perdidas_EDT_pivot_plan_opt.index, 
        #                                     columns=apertura_plan_perdidas_EDT_pivot_plan_opt.columns).fillna(0)

        for _, row_limit in cantidad_fallas_maximas_mensual_df_edt.iterrows():
            mes_limit = row_limit['Mes']
            capacidad_fallas_limit = int(row_limit['Cantidad_Fallas'])
            
            if mes_limit not in apertura_plan_perdidas_edt_opt.index:
                continue # Skip if month not in pivot table (e.g. if plan starts later)

            # Pozos active in this month with their P_EDT
            pozos_in_month_series = apertura_plan_perdidas_edt_opt.loc[mes_limit]
            pozos_in_month_series = pozos_in_month_series[pozos_in_month_series > 0] # Consider only those with loss

            # Sort: non-generics first, then by P_EDT descending
            sorted_pozos_for_month = sorted(
                pozos_in_month_series.index,
                key=lambda p_name: (not str(p_name).startswith('GENERIC_'), pozos_in_month_series[p_name]),
                reverse=True
            )
            
            pozos_to_zero_out = sorted_pozos_for_month[capacidad_fallas_limit:]
            
            for p_zero in pozos_to_zero_out:
                # pozos_eliminados_df_edt.loc[mes_limit:, p_zero] = apertura_plan_perdidas_EDT_pivot_plan_opt.loc[mes_limit:, p_zero]
                apertura_plan_perdidas_edt_opt.loc[mes_limit:, p_zero] = 0
        
        try:
            apertura_plan_perdidas_edt_opt.to_excel("Resultados/Plan_Optimizado_EDT_Final.xlsx")
            # pozos_eliminados_df_edt.to_excel("Resultados/Pozos_Eliminados_EDT_Final.xlsx")
            self.console.print("[green]Exportado 'Plan_Optimizado_EDT_Final.xlsx'.[/green]")
        except Exception as e:
            self.console.print(f"[red]Error exportando plan EDT optimizado: {e}[/red]")

        self.console.print("[green]Sección 7: Cálculo de Pérdidas EDT completado.[/green]")


    def run_section_8_calculate_cierre_upa_losses(self):
        self.console.rule("[bold blue]8. Cálculo de Pérdidas por Cierre UPA[/bold blue]")
        if not self.lista_pozos:
            self.console.print("[red]Lista de pozos no creada. Ejecute la sección 4 primero.[/red]")
            return

        pozos_con_upa_actual = [p for p in self.lista_pozos if pd.notna(p.fecha_UPA_actual)]
        
        data_cierre_upa = []
        for p_upa in pozos_con_upa_actual:
            dias_cierre_val = None
            if hasattr(p_upa, 'tiempo_cierre_actividad') and pd.notna(p_upa.tiempo_cierre_actividad):
                # If tiempo_cierre_actividad is a Series (from .loc), take first element
                dias_cierre_val = p_upa.tiempo_cierre_actividad.iloc[0] if isinstance(p_upa.tiempo_cierre_actividad, pd.Series) else p_upa.tiempo_cierre_actividad
                dias_cierre_val = pd.to_numeric(dias_cierre_val, errors='coerce')


            caudal_declino_val = p_upa.search_declino(p_upa.fecha_UPA_actual) if pd.notna(p_upa.fecha_UPA_actual) else None
            
            perdida_mensual_val = None
            if pd.notna(dias_cierre_val) and pd.notna(caudal_declino_val):
                perdida_mensual_val = dias_cierre_val * caudal_declino_val
            
            data_cierre_upa.append({
                'Nombre': p_upa.nombre,
                'Presion de Cabeza': p_upa.presion_cabeza_actual,
                'Relacion de Presion de Linea': p_upa.relacion_presion_linea,
                'Fecha Capex': p_upa.fecha_capex,
                'Fecha Crono': p_upa.fecha_crono,
                'Fecha Capex Ajustada': p_upa.fecha_capex_ajustada,
                'Estado de Actividad': p_upa.estado_actividad,
                'Estado Operativo': p_upa.estado_operativo,
                'Ultimo Control': p_upa.ultimo_control,
                'Orificio Ultimo Control': p_upa.orificio_ult_control,
                'Fecha Ultimo Control': p_upa.fecha_ultimo_control,
                'Area': p_upa.area,
                'Actividad': p_upa.actividad,
                'Campana': p_upa.campana,
                'Fecha ultima UPA': p_upa.fecha_upa_anterior,
                'Fecha UPA Actual': p_upa.fecha_UPA_actual,
                'Dias de Cierre': dias_cierre_val,
                'Caudal declino fecha UPA': caudal_declino_val,
                'Perdida mensual UPA': perdida_mensual_val
            })
        
        df_UPA_actual_perdidas_export = pd.DataFrame(data_cierre_upa)
        try:
            import os
            os.makedirs("Resultados", exist_ok=True)
            df_UPA_actual_perdidas_export.to_excel('Resultados/Curva_perdidas_cierre_UPA_actual.xlsx', index=False)
            self.console.print("[green]Exportado 'Curva_perdidas_cierre_UPA_actual.xlsx'[/green]")
        except Exception as e:
            self.console.print(f"[red]Error exportando pérdidas por cierre UPA: {e}[/red]")

        self.console.print("[green]Sección 8: Cálculo de Pérdidas por Cierre UPA completada.[/green]")


    def run_section_show_upa_monthly_report(self):
        self.console.rule("[bold blue]Reporte Mensual de Pozos en Plan UPA[/bold blue]")

        # Verificar que los dataframes de los planes UPA han sido creados
        if not hasattr(self, 'df_upa_sin_limite_export') or not hasattr(self, 'df_upa_con_limite') or self.df_upa_sin_limite_export.empty or self.df_upa_con_limite.empty:
            self.console.print("[red]Los planes UPA no han sido generados. Ejecute la Sección 5 primero.[/red]")
            return

        # Copiar los dataframes para no modificar los originales
        df_sin_limite = self.df_upa_sin_limite_export.copy()
        df_con_limite = self.df_upa_con_limite.copy()

        # --- Inicio: Recopilar datos adicionales de los objetos pozo (CORREGIDO) ---
        # Crear diccionarios para mapear la información extra de forma más robusta
        actividad_upa_map = {}
        diagnostico_map = {}
        for p in self.lista_pozos:
            actividad_upa_map[p.nombre] = p.actividad_UPA
            diagnostico_text = None
            if hasattr(p, 'diagnostico') and not p.diagnostico.empty and 'DIAGNOSTICO' in p.diagnostico.columns:
                diagnostico_text = p.diagnostico['DIAGNOSTICO'].iloc[0]
            diagnostico_map[p.nombre] = diagnostico_text

        # Añadir las nuevas columnas a los dataframes de los planes usando el mapeo
        df_sin_limite['Actividad UPA'] = df_sin_limite['Completacion'].map(actividad_upa_map)
        df_sin_limite['Diagnostico'] = df_sin_limite['Completacion'].map(diagnostico_map)
        
        df_con_limite['Actividad UPA'] = df_con_limite['Completacion'].map(actividad_upa_map)
        df_con_limite['Diagnostico'] = df_con_limite['Completacion'].map(diagnostico_map)
        # --- Fin: Recopilar datos adicionales ---

        # Asegurar que las columnas de fecha son de tipo datetime
        df_sin_limite['Fecha Capex Ajustada'] = pd.to_datetime(df_sin_limite['Fecha Capex Ajustada'], errors='coerce')
        df_con_limite['Fecha Capex Ajustada'] = pd.to_datetime(df_con_limite['Fecha Capex Ajustada'], errors='coerce')

        # Crear una columna de mes para agrupar
        df_sin_limite['Mes'] = df_sin_limite['Fecha Capex Ajustada'].dt.to_period('M')
        df_con_limite['Mes'] = df_con_limite['Fecha Capex Ajustada'].dt.to_period('M')

        # Crear un nombre de pozo para mostrar que incluya la actividad (BIF/CBM)
        df_sin_limite['display_name'] = df_sin_limite['Completacion'] + " (" + df_sin_limite['Actividad'] + ")"
        df_con_limite['display_name'] = df_con_limite['Completacion'] + " (" + df_con_limite['Actividad'] + ")"

        # --- Lógica de exportación a Excel ---
        try:
            import os
            os.makedirs("Resultados", exist_ok=True)
            
            # Definir las columnas a exportar y sus nombres finales
            cols_to_export = [
                'Mes', 'display_name', 'Actividad', 'Estado Actividad', 
                'Fecha UPA Anterior', 'Actividad UPA', 'Diagnostico'
            ]
            rename_dict = {
                'display_name': 'Pozo',
                'Estado Actividad': 'Estado de Actividad',
                'Fecha UPA Anterior': 'Fecha UPA'
            }

            # Preparar DataFrame Sin Límite
            df_export_sl = df_sin_limite.copy()
            for col in cols_to_export: # Asegurar que todas las columnas existan
                if col not in df_export_sl.columns:
                    df_export_sl[col] = None
            df_export_sl = df_export_sl[cols_to_export].rename(columns=rename_dict)

            # Preparar DataFrame Con Límite
            df_export_cl = df_con_limite.copy()
            for col in cols_to_export: # Asegurar que todas las columnas existan
                if col not in df_export_cl.columns:
                    df_export_cl[col] = None
            df_export_cl = df_export_cl[cols_to_export].rename(columns=rename_dict)

            # Ordenar y convertir el período a string para una mejor visualización en Excel
            df_export_sl = df_export_sl.sort_values('Mes').reset_index(drop=True)
            df_export_cl = df_export_cl.sort_values('Mes').reset_index(drop=True)
            df_export_sl['Mes'] = df_export_sl['Mes'].astype(str)
            df_export_cl['Mes'] = df_export_cl['Mes'].astype(str)

            output_path = 'Resultados/Reporte_Mensual_Planes_UPA.xlsx'
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                df_export_sl.to_excel(writer, sheet_name='Plan Sin Límite', index=False)
                df_export_cl.to_excel(writer, sheet_name='Plan Con Límite', index=False)
            
            self.console.print(f"\n[green]Reporte mensual exportado exitosamente a '{output_path}'[/green]")
        except Exception as e:
            self.console.print(f"\n[red]Error al exportar el reporte a Excel: {e}[/red]")
        # --- Fin de la lógica de exportación ---

        # Agrupar por mes y obtener la lista de pozos para la tabla de consola
        sin_limite_grouped = df_sin_limite.groupby('Mes')['display_name'].apply(list).to_dict()
        con_limite_grouped = df_con_limite.groupby('Mes')['display_name'].apply(list).to_dict()

        # Definir el rango de meses para el reporte (ej. para el año 2025)
        rango_meses_reporte = pd.date_range(start='2025-01-01', end='2025-12-31', freq='MS').to_period('M')

        # Crear y mostrar la tabla del reporte en la consola
        table = Table(title="Reporte Mensual de Pozos en Ventana (BIF/CBM)", show_lines=True)
        table.add_column("Mes", style="cyan", no_wrap=True)
        table.add_column("Plan SIN Límite (Pozos)", style="magenta", overflow="fold")
        table.add_column("Plan CON Límite (Pozos)", style="green", overflow="fold")

        for mes in rango_meses_reporte:
            pozos_sl = sin_limite_grouped.get(mes, [])
            pozos_cl = con_limite_grouped.get(mes, [])
            
            pozos_sl_str = "\n".join(pozos_sl) if pozos_sl else "[italic]-- Sin pozos --[/italic]"
            pozos_cl_str = "\n".join(pozos_cl) if pozos_cl else "[italic]-- Sin pozos --[/italic]"
            
            table.add_row(mes.strftime('%Y-%m'), pozos_sl_str, pozos_cl_str)

        self.console.print(table)
        self.console.print("\n[green]Reporte mensual de planes UPA completado.[/green]")

def conectarse_cns(query):
    """
    Se conecta a Oracle CNS, ejecuta la consulta y devuelve un DataFrame.
    """
    try:
        user_id = 'sahara'    
        user_password = 'sahara'
        server = 'SLPBUETBORA15'    
        port = 1527    
        sid = "PSSH" 
        
        dsn_tns = cx_Oracle.makedsn(server, port, sid)
        connection=cx_Oracle.connect(user=user_id, password=user_password, dsn=dsn_tns)
        
        df = pd.read_sql(query, connection)
        return df
    except Exception as e:
        print(f"Error en conectarse_cns: {e}")
        return pd.DataFrame() # Devuelve un DataFrame vacío en caso de error

def conectarse_teradata(query, chunksize=None):
    """
    Se conecta a Teradata. Si se proporciona un chunksize, lee los datos en bloques
    y muestra el progreso. De lo contrario, lee todo de una vez.
    """
    try:
        dict_conection = {
            'host': 'tdprod',
            'logmech': 'LDAP',
            'user': 'ry32287',
            'password': '4theEmperor!!'
        }
        with teradatasql.connect(**dict_conection) as conexion:
            if chunksize:
                # Modo de lectura en bloques con barra de progreso
                console = Console()
                console.print("[italic]Modo de lectura en bloques activado.[/italic]")
                
                # Usamos un cursor para ejecutar la consulta y obtener los nombres de las columnas
                cursor = conexion.cursor()
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                
                all_chunks = []
                total_rows = 0
                
                # Usamos un bucle para ir pidiendo los bloques de datos
                with tqdm(desc="Recibiendo filas", unit=" filas") as pbar:
                    while True:
                        chunk_data = cursor.fetchmany(chunksize)
                        if not chunk_data:
                            break # Se terminaron los datos
                        all_chunks.extend(chunk_data)
                        total_rows += len(chunk_data)
                        pbar.update(len(chunk_data)) # Actualiza la barra de progreso
                
                console.print(f"[green]Total de {total_rows} filas recibidas. Creando DataFrame...[/green]")
                df = pd.DataFrame(all_chunks, columns=columns)
            else:
                # Modo de lectura normal (para las consultas rápidas)
                df = pd.read_sql(query, conexion)
        return df
    except Exception as e:
        print(f"Error en conectarse_teradata: {e}")
        return pd.DataFrame() # Devuelve un DataFrame vacío en caso de error


def main_menu():
    workflow = UPAWorkflow()
    console = Console()

    menu_actions = {
        "1": workflow.run_section_1_load_excel_data,
        "1B": workflow.run_section_1b_load_from_database, # Nueva opción
        "2": workflow.run_section_2_save_to_parquet,
        "3": workflow.run_section_3_load_from_parquet,
        "A": workflow.run_section_show_parquet_df_info, # Nueva opción
        "4": workflow.run_section_4_create_pozo_objects,
        "B": workflow.run_section_inspect_pozo,
        "5": workflow.run_section_5_create_upa_plans,
        "6": workflow.run_section_6_process_aseguramiento,
        "7": workflow.run_section_7_calculate_edt_losses,
        "8": workflow.run_section_8_calculate_cierre_upa_losses,
        "C": workflow.run_section_show_upa_monthly_report, # Nueva acción
    }

    while True:
        console.rule("[bold cyan]Menú Principal de Procesamiento UPA[/bold cyan]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Opción", style="dim", width=6)
        table.add_column("Descripción")

        table.add_row("1", "Cargar y Procesar Datos desde [bold yellow]Excel[/bold yellow]")
        table.add_row("1B", "Cargar y Procesar Datos desde [bold green]Bases de Datos[/bold green] (CNS/Teradata)")
        table.add_row("2", "Guardar DataFrames a Parquet")
        table.add_row("3", "Cargar DataFrames desde Parquet")
        table.add_row("A", "Mostrar Información de DataFrames Parquet Cargados")
        table.add_row("4", "Crear Objetos Pozo")
        table.add_row("B", "Inspeccionar un Objeto Pozo Específico")
        table.add_row("5", "Crear Planes UPA (Sin Límite y Con Límite)")
        table.add_row("C", "Ver Reporte Mensual de Planes UPA")
        table.add_row("6", "Procesar Aseguramiento de Pozos")
        table.add_row("7", "Calcular Pérdidas EDT")
        table.add_row("8", "Calcular Pérdidas por Cierre UPA")
        table.add_row("9", "Ejecutar Todas las Secciones (1-8, excluye A, B, C)")
        table.add_row("9B", "Ejecutar Todas las Secciones desde [bold green]BD[/bold green] (1B, 2-8, excluye A, B, C)")
        table.add_row("0", "Salir")
        
        console.print(table)
        choice = Prompt.ask("Seleccione una opción", choices=list(menu_actions.keys()) + ["9", "9B", "0"], default="0")
        if choice == "0":
            console.print("[bold yellow]Saliendo del programa.[/bold yellow]")
            break
        elif choice == "9":
            console.print("[bold yellow]Ejecutando todas las secciones secuencialmente (desde Excel)...[/bold yellow]")
            # Ejecuta secciones 1-3 y luego 4-8
            sections_to_run = ["1", "2", "3", "4", "5", "6", "7", "8"]
            for section_key in sections_to_run:
                action = menu_actions.get(section_key)
                if action:
                    try:
                        console.print(f"[bold blue]Ejecutando Sección {section_key}...[/bold blue]")
                        action()
                    except Exception as e:
                        console.print(f"[bold red]Error en la sección {section_key}: {e}[/bold red]")
                        console.print_exception(show_locals=True)
                        break 
            console.print("[bold green]Todas las secciones ejecutadas (o detenidas por error).[/bold green]")
        elif choice == "9B":
            console.print("[bold yellow]Ejecutando todas las secciones secuencialmente (desde Bases de Datos)...[/bold yellow]")
            # Ejecuta secciones 1B, 2-3 y luego 4-8
            sections_to_run = ["1B", "2", "3", "4", "5", "6", "7", "8"]
            for section_key in sections_to_run:
                action = menu_actions.get(section_key)
                if action:
                    try:
                        console.print(f"[bold blue]Ejecutando Sección {section_key}...[/bold blue]")
                        action()
                    except Exception as e:
                        console.print(f"[bold red]Error en la sección {section_key}: {e}[/bold red]")
                        console.print_exception(show_locals=True)
                        break 
            console.print("[bold green]Todas las secciones ejecutadas (o detenidas por error).[/bold green]")
        elif choice in menu_actions:
            action = menu_actions[choice]
            try:
                action()
            except Exception as e:
                console.print(f"[bold red]Error ejecutando la sección {choice}: {e}[/bold red]")
                console.print_exception(show_locals=True)
        else:
            console.print("[red]Opción no válida. Intente de nuevo.[/red]")
        
        Prompt.ask("\n[cyan]Presione Enter para continuar...[/cyan]", default="")

if __name__ == "__main__":
    main_menu()

    # Todo

    # Armar la para el analisis RTA
    # Para realizar la prueba del commit