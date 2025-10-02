<h1 style="text-align: center;">TP SVM - Support Vector Machines</h1>

Bienvenue sur la page source du projet **TP SVM** !  

---

##  Description du projet

Ce projet a été réalisé dans le cadre du master **Statistiques et Science des Données** et du master **Management de l'Information et de la Décision** à l’Université de Montpellier.  
L’objectif principal est de comprendre et expérimenter les **Support Vector Machines (SVM)** à travers différents jeux de données et contextes d’application.  

Les travaux combinent des expériences pratiques en Python et une analyse théorique documentée dans un rapport LaTeX.  

---

##  Objectifs pédagogiques

- Manipuler et comprendre les SVM à noyau linéaire et polynomial.  
- Étudier l’influence des hyperparamètres `C` et `γ`.  
- Observer les comportements d’un SVM sur données équilibrées et déséquilibrées.  
- Expérimenter l’ajout de variables de nuisance et leur impact sur la performance.  
- Améliorer la généralisation via une réduction de dimension par **PCA**.  
- Appliquer les SVM sur un jeu de données réel de **reconnaissance de visages** (LFW dataset).  

---

##  Arborescence du projet

```text
TP_SVM/
├── code/                   
│   ├── svm_gui.py       
│   ├── svm_script.py                
│   └── svm_source.py        
│
├── rapport/                
│   ├── rapport.pdf            
│   ├── rapport.tex       
│   └── images/             
│       ├── gauss1.png
│       ├── gauss2.png
│       ├── linC=0.01.png
│       ├── linC=1.png
│       ├── linC=7.png
│       └── ...
│
├── requirements.txt        
├── README.md               
└── LICENSE               
```

---

## Installation et exécution

Ce dépôt contient :

un script Python dans code/svm_script.py ;

un rapport rédigé en LaTeX dans rapport/rapport.tex.

1. Cloner le projet :

Sous WSL, commencez par cloner le projet et entrez dans le dossier :

git clone https://github.com/ksarih/TP_SVM.git
cd TP_SVM

2. Installer les dépendances :

pip install -r requirements.txt

3. Lancer le code Python :

cd code
python svm_script.py

4. Compiler le rapport LaTeX :

cd rapport
latexmk -pdf -shell-escape rapport.tex


Le fichier rapport.pdf sera généré dans le dossier rapport/.

---

## Auteurs

- [**Kaoutar SARIH**](https://github.com/ksarih)  
- [**Doha EL QEMMAH**](https://github.com/elqemmahdoha)