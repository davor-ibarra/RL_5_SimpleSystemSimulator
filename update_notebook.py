import nbformat as nbf
import os

nb_path = r"c:\Users\davor\Escritorio\DOCTORADO\Tesis\RL_5_SimpleSystemSimulator\1_notebook_sanityCheck_episodio_v1.ipynb"

# Load the notebook
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Plot for Cart Position parameters
code_cart_reward = """y_cols_cart = ['L_e_cart_position', 'L_edot_cart_position', 'L_delta_u_cart_position', 'reward_kp_cart_position']

x = interval_timeline_df['interval_id'].to_numpy()
ax = interval_timeline_df.plot(x='interval_id', y=y_cols_cart, figsize=(10, 3), grid=True)

ax.set_xlabel('Interval')
ax.set_title('Reward Components - Cart Position')
plt.tight_layout()
plt.show()"""
nb.cells.append(nbf.v4.new_code_cell(code_cart_reward))

# Plot for Global Interval Reward
code_global_reward = """# Recompensa global por intervalo
ax = interval_timeline_df.plot(x='interval_id', y='global_interval_reward', figsize=(10, 3), grid=True, color='purple')
ax.set_xlabel('Interval')
ax.set_title('Global Interval Reward')
plt.tight_layout()
plt.show()"""
nb.cells.append(nbf.v4.new_code_cell(code_global_reward))


# New Section: Análisis de Episodios
md_analysis = """## Caja 7 - Análisis de Episodios (Global)
- Extraemos las métricas finales (`end_episode_data`) de todos los episodios iterando sobre los chunks encontrados.
- Generamos un DataFrame global para ver la evolución del entrenamiento y del Reward."""
nb.cells.append(nbf.v4.new_markdown_cell(md_analysis))

# Load global episodes data
code_global_load = """# Cargar un resumen de todos los episodios
global_records = []

for cf in chunk_files:
    try:
        chunk_data = load_json(cf)
        for ep in chunk_data:
            ep_id = extract_episode_id(ep)
            end_data = ep.get('end_episode_data', {})
            
            record = {'episode_id': ep_id}
            if end_data:
                record.update(end_data)
            
            global_records.append(record)
    except Exception as e:
        print(f"Error cargando {cf}: {e}")

global_df = pd.DataFrame(global_records).sort_values('episode_id').reset_index(drop=True)
print(f"Total episodios cargados: {len(global_df)}")
global_df.head()"""
nb.cells.append(nbf.v4.new_code_cell(code_global_load))

# Plot global training metrics
code_global_plot = """# Gráfico de evolución del Total Reward a lo largo de los episodios
fig, ax1 = plt.subplots(figsize=(12, 4))

ax1.plot(global_df['episode_id'], global_df['total_reward'], color='tab:blue', alpha=0.6, label='Total Reward')
ax1.set_xlabel('Episode ID')
ax1.set_ylabel('Total Reward', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, alpha=0.3)

# Epsilon en otro eje Y, si está presente
if 'epsilon' in global_df.columns:
    ax2 = ax1.twinx()
    ax2.plot(global_df['episode_id'], global_df['epsilon'], color='tab:orange', linewidth=2, label='Epsilon')
    ax2.set_ylabel('Epsilon', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

plt.title('Evolución del Reward Total y Epsilon por Episodio')
fig.tight_layout()
plt.show()"""
nb.cells.append(nbf.v4.new_code_cell(code_global_plot))

code_termination = """# Razón de terminación de los episodios (estadísticas)
if 'end_termination_reason' in global_df.columns:
    term_counts = global_df['end_termination_reason'].value_counts()
    print("Razones de terminación de episodio:")
    print(term_counts)
    
    # Gráfico de torta
    term_counts.plot.pie(autopct='%1.1f%%', figsize=(5,5))
    plt.title('Termination Reasons')
    plt.ylabel('')
    plt.show()"""
nb.cells.append(nbf.v4.new_code_cell(code_termination))

# Save the updated notebook
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook updated successfully.")
