#!/usr/bin/env python3
"""
SmartPlay AI — Extraction de statistiques de match et génération de table LaTeX pour PFE.
Ce script lit 'data.csv' et 'summary.json' dans le répertoire courant pour calculer
les statistiques clés des joueurs et générer un fichier LaTeX prêt à être inséré dans le rapport.
"""

import os
import json
import pandas as pd
import numpy as np

def calculate_stats(csv_path="data.csv", summary_path="summary.json"):
    print(f"[*] Lecture du fichier de télémétrie : {csv_path}")
    if not os.path.exists(csv_path):
        print(f"[ERR] Le fichier {csv_path} n'existe pas.")
        return None, None

    # Chargement du DataFrame
    df = pd.read_csv(csv_path)
    
    # Nettoyage des types de colonnes
    for col in df.columns:
        if col != "frame":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # Chargement du summary.json pour le nombre de frappes
    player_shoots = {"1": 0, "2": 0, "3": 0, "4": 0}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
                player_shoots = summary_data.get("player_shoots", player_shoots)
                print(f"[+] Nombre de frappes chargé depuis {summary_path}")
        except Exception as e:
            print(f"[WARN] Erreur lors de la lecture de {summary_path}: {e}")
            
    stats = {}
    
    # Calcul des métriques pour chaque joueur (1 à 4)
    for pid in range(1, 5):
        stats[pid] = {}
        
        # Colonnes clés
        vcol = f"player{pid}_Vnorm4"
        dcol = f"player{pid}_distance"
        ycol = f"player{pid}_y"
        
        # 1. Distance totale
        total_dist = 0.0
        if dcol in df.columns:
            total_dist = float(df[dcol].fillna(0).sum())
        stats[pid]["distance_m"] = round(total_dist, 1)
        
        # 2. Vitesse moyenne et maximale
        avg_speed = 0.0
        max_speed = 0.0
        speed_std = 0.0
        
        # Utiliser Vnorm4 ou Vnorm1 si Vnorm4 n'existe pas
        if vcol not in df.columns:
            vcol = f"player{pid}_Vnorm1"
            
        if vcol in df.columns:
            # Filtrer les valeurs aberrantes physiques (supérieures à 25 km/h ≈ 6.94 m/s)
            clean_v = df[vcol].dropna().abs()
            clean_v = clean_v[clean_v < 6.94]
            
            if not clean_v.empty:
                avg_speed = round(float(clean_v.mean()) * 3.6, 1)
                max_speed = round(float(clean_v.quantile(0.98)) * 3.6, 1)
                speed_std = round(float(clean_v.std()) * 3.6, 1)
                
        stats[pid]["avg_speed_kmh"] = avg_speed
        stats[pid]["max_speed_kmh"] = max_speed
        stats[pid]["speed_consistency_kmh"] = speed_std
        
        # 3. Dépense calorique (0.14 kcal par mètre)
        stats[pid]["calorie_burn_kcal"] = round(total_dist * 0.14, 1)
        
        # 4. Présence au filet (% de frames à moins de 3 mètres du filet)
        net_presence_pct = 0.0
        if ycol in df.columns:
            valid_y = df[ycol].dropna()
            if not valid_y.empty:
                net_presence_pct = round(float((valid_y.abs() < 3.0).sum() / len(valid_y)) * 100, 1)
        stats[pid]["net_presence_pct"] = net_presence_pct
        
        # 5. Nombre de frappes
        stats[pid]["shoots"] = int(player_shoots.get(str(pid), 0))
        
    return stats, df

def generate_latex_table(stats, output_path="match_stats_table.tex"):
    if not stats:
        return
        
    latex_code = r"""% Table générée automatiquement par SmartPlay AI - Extraction Télémétrie
\begin{table}[htbp]
  \centering
  \caption{Métriques de Performance Spatiotemporelles et Tactiques Individuelles}
  \label{tab:match_stats_players}
  \tablestyle
  \begin{tabularx}{\textwidth}{X C{2.2cm} C{2.2cm} C{2.2cm} C{2.2cm}}
    \toprule
    \rowcolor{tablehead}
    \textcolor{white}{\textbf{Indicateur Physique \& Tactique (KPI)}} & 
    \textcolor{white}{\textbf{Joueur 1}} & 
    \textcolor{white}{\textbf{Joueur 2}} & 
    \textcolor{white}{\textbf{Joueur 3}} & 
    \textcolor{white}{\textbf{Joueur 4}} \\
    \midrule
"""
    
    # Lignes de données
    kpis = [
        ("Distance parcourue (m)", "distance_m", "{:.1f}"),
        ("Vitesse moyenne (km/h)", "avg_speed_kmh", "{:.1f}"),
        ("Vitesse maximale (km/h)", "max_speed_kmh", "{:.1f}"),
        (r"Régularité de vitesse ($\sigma$ en km/h)", "speed_consistency_kmh", "{:.1f}"),
        ("Dépense énergétique (kcal)", "calorie_burn_kcal", "{:.1f}"),
        (r"Présence offensive au filet (\%)", "net_presence_pct", "{:.1f}\\%"),
        ("Nombre de frappes (Strikes)", "shoots", "{:d}")
    ]
    
    for idx, (label, key, fmt) in enumerate(kpis):
        row_color = "rowalt" if idx % 2 == 0 else "rowalt2"
        line = f"    \\rowcolor{{{row_color}}}\n    \\textbf{{{label}}} "
        for pid in range(1, 5):
            val = stats[pid][key]
            line += f"& {fmt.format(val)} "
        line += "\\\\\n"
        latex_code += line
        
    latex_code += r"""    \bottomrule
  \end{tabularx}
\end{table}
"""

    # Enregistrer dans le répertoire courant
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(f"[+] Fichier LaTeX généré avec succès : {output_path}")

    # Enregistrer également dans le répertoire du rapport LaTeX
    latex_project_dir = r"c:\Users\dell\Downloads\pfe_rapport-main\pfe_rapport-main"
    if os.path.exists(latex_project_dir):
        dest_path = os.path.join(latex_project_dir, "match_stats_table.tex")
        try:
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(latex_code)
            print(f"[+] Fichier LaTeX copié avec succès dans le projet de rapport : {dest_path}")
        except Exception as e:
            print(f"[WARN] Impossible d'écrire dans {dest_path}: {e}")


def print_terminal_table(stats):
    if not stats:
        return
    
    print("\n" + "="*85)
    print("                      RÉCAPITULATIF DES MÉTRIQUES EXTRAITES")
    print("="*85)
    print(f"{'Métrique (KPI)':<40} | {'Joueur 1':<8} | {'Joueur 2':<8} | {'Joueur 3':<8} | {'Joueur 4':<8}")
    print("-"*85)
    
    kpis = [
        ("Distance parcourue (m)", "distance_m", ".1f"),
        ("Vitesse moyenne (km/h)", "avg_speed_kmh", ".1f"),
        ("Vitesse maximale (km/h)", "max_speed_kmh", ".1f"),
        ("Régularité de vitesse (std km/h)", "speed_consistency_kmh", ".1f"),
        ("Dépense énergétique (kcal)", "calorie_burn_kcal", ".1f"),
        ("Présence offensive au filet (%)", "net_presence_pct", ".1f"),
        ("Nombre de frappes (Strikes)", "shoots", "d")
    ]
    
    for label, key, fmt in kpis:
        row = f"{label:<40} | "
        for pid in range(1, 5):
            val = stats[pid][key]
            formatted_val = f"{val:{fmt}}"
            row += f"{formatted_val:<8} | "
        print(row[:-3])
        
    print("="*85)

def main():
    csv_file = "data.csv"
    
    # Essaye de trouver un data_new.csv si disponible
    if os.path.exists("data_new.csv"):
        csv_file = "data_new.csv"
        
    stats, _ = calculate_stats(csv_path=csv_file)
    if stats:
        print_terminal_table(stats)
        generate_latex_table(stats)

if __name__ == "__main__":
    main()
