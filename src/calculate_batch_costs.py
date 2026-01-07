#%%
import json

# --- Precios por cada millón de tokens ---
PRECIO_INPUT_POR_MILLON = 0.25/2  # $0.25
PRECIO_OUTPUT_POR_MILLON = 2.00/2  # $2.00

def sumar_tokens_de_jsonl(ruta_archivo):
    """
    Lee un archivo JSONL, extrae los input_tokens y output_tokens de cada línea
    y calcula la suma total de cada uno.

    Args:
        ruta_archivo (str): La ruta al archivo .jsonl a procesar.

    Returns:
        tuple: Una tupla conteniendo la suma total de input_tokens y output_tokens.
    """
    total_input_tokens = 0
    total_output_tokens = 0

    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            for linea in f:
                try:
                    # Cargar cada línea como un objeto JSON
                    dato = json.loads(linea)

                    # Acceder a los campos anidados de manera segura
                    usage = dato.get("response", {}).get("body", {}).get("usage", {})
                    if usage:
                        total_input_tokens += usage.get("input_tokens", 0)
                        total_output_tokens += usage.get("output_tokens", 0)

                except json.JSONDecodeError:
                    print(f"Advertencia: Se omitió una línea que no es un JSON válido: {linea.strip()}")
                except AttributeError:
                    print(f"Advertencia: Se omitió una línea con una estructura inesperada: {linea.strip()}")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {ruta_archivo}")
        return None, None

    return total_input_tokens, total_output_tokens

def calcular_costo_total(total_input_tokens, total_output_tokens):
    """
    Calcula el costo total basado en la cantidad de tokens de entrada y salida.

    Args:
        total_input_tokens (int): La cantidad total de tokens de entrada.
        total_output_tokens (int): La cantidad total de tokens de salida.

    Returns:
        tuple: Una tupla con el costo de input, costo de output y el costo total.
    """
    costo_input = (total_input_tokens / 1_000_000) * PRECIO_INPUT_POR_MILLON
    costo_output = (total_output_tokens / 1_000_000) * PRECIO_OUTPUT_POR_MILLON
    costo_total = costo_input + costo_output
    
    return costo_input, costo_output, costo_total

def analizar_cambio_en_tokens(ruta_archivo, n_request):
    """
    Analiza si hubo un cambio en el promedio de tokens por request a partir de una request 'n'.

    Args:
        ruta_archivo (str): La ruta al archivo .jsonl a procesar.
        n_request (int): El número de la request (línea) a partir de la cual se quiere comparar.
                         La comparación se hará entre (1 a n-1) y (n hasta el final).
    """
    # Acumuladores para el periodo "antes" de n
    input_antes, output_antes, count_antes = 0, 0, 0
    # Acumuladores para el periodo "después" (incluyendo n)
    input_despues, output_despues, count_despues = 0, 0, 0
    
    linea_actual = 0

    print("\n" + "="*40)
    print(f"Análisis de Cambio de Tokens en la Request #{n_request}")
    print("="*40)

    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            for linea in f:
                linea_actual += 1
                try:
                    dato = json.loads(linea)
                    usage = dato.get("response", {}).get("body", {}).get("usage", {})
                    
                    if usage:
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                        
                        if linea_actual < n_request:
                            input_antes += input_tokens
                            output_antes += output_tokens
                            count_antes += 1
                        else:
                            input_despues += input_tokens
                            output_despues += output_tokens
                            count_despues += 1
                
                except (json.JSONDecodeError, AttributeError):
                    # Omitimos las líneas con errores, ya se advirtió en la función principal
                    continue

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {ruta_archivo}")
        return

    # --- Calcular promedios y mostrar resultados ---
    
    # Periodo 1: Antes de la request n
    print(f"--- Periodo 1: Requests 1 a {n_request - 1} ({count_antes} en total) ---")
    if count_antes > 0:
        avg_input_antes = input_antes / count_antes
        avg_output_antes = output_antes / count_antes
        avg_total_antes = (input_antes + output_antes) / count_antes
        print(f"Promedio Input Tokens:  {avg_input_antes:,.2f}")
        print(f"Promedio Output Tokens: {avg_output_antes:,.2f}")
        print(f"Promedio Total Tokens:  {avg_total_antes:,.2f}")

        avg_input_cost_antes = (avg_input_antes / 1_000_000) * PRECIO_INPUT_POR_MILLON
        avg_output_cost_antes = (avg_output_antes / 1_000_000) * PRECIO_OUTPUT_POR_MILLON
        avg_total_cost_antes = avg_input_cost_antes + avg_output_cost_antes
        print(f"Promedio Input Cost:  ${avg_input_cost_antes:,.5f}")
        print(f"Promedio Output Cost: ${avg_output_cost_antes:,.5f}")
        print(f"Promedio Total Cost:  ${avg_total_cost_antes:,.5f}")

    else:
        print("No hay requests en este periodo para analizar.")
        avg_total_antes = 0 # Para evitar errores en el cálculo de cambio

    # Periodo 2: Desde la request n hasta el final
    print(f"\n--- Periodo 2: Requests {n_request} en adelante ({count_despues} en total) ---")
    if count_despues > 0:
        avg_input_despues = input_despues / count_despues
        avg_output_despues = output_despues / count_despues
        avg_total_despues = (input_despues + output_despues) / count_despues
        print(f"Promedio Input Tokens:  {avg_input_despues:,.2f}")
        print(f"Promedio Output Tokens: {avg_output_despues:,.2f}")
        print(f"Promedio Total Tokens:  {avg_total_despues:,.2f}")

        avg_input_cost_despues = (avg_input_despues / 1_000_000) * PRECIO_INPUT_POR_MILLON
        avg_output_cost_despues = (avg_output_despues / 1_000_000) * PRECIO_OUTPUT_POR_MILLON
        avg_total_cost_despues = avg_input_cost_despues + avg_output_cost_despues
        print(f"Promedio Input Cost:  ${avg_input_cost_despues:,.5f}")
        print(f"Promedio Output Cost: ${avg_output_cost_despues:,.5f}")
        print(f"Promedio Total Cost:  ${avg_total_cost_despues:,.5f}")
    else:
        print("No hay requests en este periodo para analizar.")
        avg_total_despues = 0

    # --- Comparativa y Conclusión ---
    if count_antes > 0 and count_despues > 0:
        print("\n--- Comparativa de Promedios ---")
        cambio_porcentual = ((avg_total_despues - avg_total_antes) / avg_total_antes) * 100
        print(f"Cambio en tokens totales por request: {cambio_porcentual:+.2f}%")
        if cambio_porcentual > 5:
            print("Conclusión: Hubo un AUMENTO significativo en el consumo de tokens.")
        elif cambio_porcentual < -5:
            print("Conclusión: Hubo una DISMINUCIÓN significativa en el consumo de tokens.")
        else:
            print("Conclusión: El consumo de tokens por request se mantuvo relativamente estable.")
    print("="*40 + "\n")

# <--- FIN DE LA NUEVA FUNCIÓN --->


# --- Uso del script ---
# Reemplaza 'tu_archivo.jsonl' con la ruta real a tu archivo.
ruta_del_archivo = '/home/jcuello/emotion_drift/data/04_annotated/annotated_results.jsonl'
input_tokens, output_tokens = sumar_tokens_de_jsonl(ruta_del_archivo)

if input_tokens is not None and output_tokens is not None:
    # Imprimir los totales de tokens
    print("--- Resumen de Tokens ---")
    print(f"Cantidad total de input_tokens:  {input_tokens:,}".replace(',', '.'))
    print(f"Cantidad total de output_tokens: {output_tokens:,}".replace(',', '.'))
    print("-" * 25)

    # Calcular y mostrar los costos
    costo_input, costo_output, costo_total = calcular_costo_total(input_tokens, output_tokens)
    
    print("--- Desglose de Costos (USD) ---")
    # Usamos :.6f para mostrar decimales, útil para costos pequeños
    print(f"Costo de Input:  ${costo_input:.6f}")
    print(f"Costo de Output: ${costo_output:.6f}")
    print("-" * 25)
    print(f"COSTO TOTAL:     ${costo_total:.6f}")

    # Define en qué request quieres hacer la comprobación.
    request_de_cambio = 15203
    analizar_cambio_en_tokens(ruta_del_archivo, request_de_cambio)
# %%
