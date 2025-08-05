# hook_store.py

"""
Un simple almacén global para guardar las activaciones capturadas.
Como los módulos de Python son singletons, este diccionario será compartido
entre el proceso principal y el proceso trabajador de vLLM.
"""

# Este es el diccionario que compartiremos
ACTIVATION_HOOKS = {}