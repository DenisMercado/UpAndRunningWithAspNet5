import pandas as pd
import numpy as np
import re
import os


def _normalize_well_name(name):
    """
    Limpia y normaliza un nombre de pozo para facilitar las comparaciones.
    Esta función es consistente con la usada en el script principal UPA_APP_3.py.
    """
    if not isinstance(name, str):
        return ""
    # Convierte a minúsculas, quita espacios y prefijos/sufijos comunes
    normalized = str(name).lower().strip()
    if normalized.startswith('ypf.nq.'):
        normalized = normalized[len('ypf.nq.'):]
    if normalized.endswith('(h)'):
        normalized = normalized[:-3]
    # Elimina corchetes y su contenido, por ejemplo [00]a, [01], []
    normalized = re.sub(r'\[.*?\]', '', normalized)
    # Elimina ceros a la izquierda después del guion (ej: lach-001 -> lach-1)
    normalized = re.sub(r'-(0+)(\d+)', r'-\2', normalized)
    # Elimina espacios extra que puedan quedar tras quitar corchetes
    normalized = normalized.strip()
    return normalized


# --- 1. Carga de Datos ---
print("Cargando archivos...")
try:
    archivo_declinos = r"C:\Users\RY32287\OneDrive - YPF\Documentos\Python Scripts\UPA\datos_prueba\NOC_G&R_PERFIL_UPA_DECLINO.xlsx"
    declinos = pd.read_excel(archivo_declinos, sheet_name='Sheet1', engine='openpyxl')

    archivo_completaciones = r"C:\Users\RY32287\OneDrive - YPF\Documentos\Python Scripts\UPA\datos_procesados\UPS_DIM_COMPLETACION.parquet"
    maestros_pozos = pd.read_parquet(archivo_completaciones)
    print("Archivos cargados correctamente.")
except FileNotFoundError as e:
    print(f"Error: No se pudo encontrar el archivo {e.filename}")
    exit()

# --- Pre-procesamiento de Declinos ---
# Asegurar que la columna de fecha sea datetime
declinos['FECHA'] = pd.to_datetime(declinos['FECHA'], errors='coerce')
# Calcular BRUTA si no existe, asegurando que las columnas base son numéricas
declinos['PETRÓLEO_(m3/DC)'] = pd.to_numeric(declinos['PETRÓLEO_(m3/DC)'], errors='coerce').fillna(0)
declinos['AGUA_(m3/DC)'] = pd.to_numeric(declinos['AGUA_(m3/DC)'], errors='coerce').fillna(0)
declinos['BRUTA_(m3/DC)'] = declinos['PETRÓLEO_(m3/DC)'] + declinos['AGUA_(m3/DC)']

# --- 2. Normalización de Nombres de Pozos ---
print("\nNormalizando nombres de pozos para comparación...")

# Limpiar nombres en el archivo de declinos
declinos['POZO'] = declinos['POZO'].str.replace('YPF.Nq.', '', regex=False).str.strip()

# Crear columnas normalizadas en ambos DataFrames
maestros_pozos['_norm_name'] = maestros_pozos['Completacion_Nombre_Corto_Modificado'].apply(_normalize_well_name)
declinos['_norm_name'] = declinos['POZO'].apply(_normalize_well_name)

# --- 3. Identificación y Agregado de Pozos Faltantes ---
print("Identificando pozos con producción en el archivo de declinos...")
# Filtrar para obtener solo los pozos que tienen al menos un registro de producción > 0
pozos_con_produccion_en_declino = declinos[declinos['BRUTA_(m3/DC)'] > 0]['_norm_name'].unique()
nombres_declino_con_produccion = set(pozos_con_produccion_en_declino)
print(f"Se encontraron {len(nombres_declino_con_produccion)} pozos con registros de producción en 'declinos'.")

nombres_maestro = set(maestros_pozos['_norm_name'])
# La comparación ahora se hace contra los pozos que sí tienen producción
pozos_faltantes_norm = nombres_declino_con_produccion - nombres_maestro

maestros_pozos_actualizado = maestros_pozos.copy()

if pozos_faltantes_norm:
    print(f"\nSe encontraron {len(pozos_faltantes_norm)} pozos en 'declinos' que no están en 'maestros_pozos'.")

    # Obtener los nombres originales de los pozos a agregar
    pozos_a_agregar_df = declinos[declinos['_norm_name'].isin(pozos_faltantes_norm)][['POZO', '_norm_name']].drop_duplicates()

    nuevos_pozos_lista = []
    for _, row in pozos_a_agregar_df.iterrows():
        pozo_nombre_original = row['POZO']
        pozo_nombre_norm = row['_norm_name']

        # Buscar la primera fecha de producción (Fecha PEM)
        # 1. Filtrar el declino para el pozo actual
        declino_del_pozo = declinos[declinos['_norm_name'] == pozo_nombre_norm]
        # 2. Encontrar las filas donde la producción bruta fue mayor a cero
        produccion_positiva = declino_del_pozo[declino_del_pozo['BRUTA_(m3/DC)'] > 0]

        fecha_pem = pd.NaT # Valor por defecto si no se encuentra producción
        if not produccion_positiva.empty:
            # 3. Ordenar por fecha y tomar la primera
            fecha_pem = produccion_positiva.sort_values(by='FECHA').iloc[0]['FECHA']

        nuevos_pozos_lista.append({
            'Completacion_Nombre_Corto_Modificado': pozo_nombre_original,
            'Metodo_Produccion_Actual_Cd': 'FA', # Se completa con 'FA' como solicitado
            'Fecha_Inicio_Produccion_Dt': fecha_pem
        })

    nuevos_pozos_df = pd.DataFrame(nuevos_pozos_lista)

    # Concatenar con el maestro original. Pandas alineará las columnas y llenará con NaN donde no haya datos.
    maestros_pozos_actualizado = pd.concat([maestros_pozos, nuevos_pozos_df], ignore_index=True)

    print("\nSe han agregado los siguientes pozos al maestro con la información completada:")
    print(nuevos_pozos_df[['Completacion_Nombre_Corto_Modificado', 'Metodo_Produccion_Actual_Cd', 'Fecha_Inicio_Produccion_Dt']].to_string(index=False))
    print("\nNOTA: El resto de los datos (Área, etc.) están vacíos (NaN) y deben ser completados si es necesario.")
else:
    print("\n¡Excelente! El archivo maestro de pozos ya contiene todos los pozos presentes en el archivo de declinos.")

# --- 4. Guardado del Resultado ---
# Limpiar las columnas temporales antes de guardar
maestros_pozos_actualizado.drop(columns=['_norm_name'], inplace=True, errors='ignore')
declinos.drop(columns=['_norm_name'], inplace=True, errors='ignore')

output_dir = r"C:\Users\RY32287\OneDrive - YPF\Documentos\Python Scripts\UPA\datos_procesados"
os.makedirs(output_dir, exist_ok=True)
archivo_salida = os.path.join(output_dir, "UPS_DIM_COMPLETACION_actualizado.parquet")

maestros_pozos_actualizado.to_parquet(archivo_salida, index=False)

print(f"\nProceso finalizado. Maestro de pozos actualizado y guardado en:\n{archivo_salida}")
print(f"Shape del maestro original:   {maestros_pozos.shape}")
print(f"Shape del maestro actualizado: {maestros_pozos_actualizado.shape}")
