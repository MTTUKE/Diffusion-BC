# Diffusion Model for Coronal Hole Segmentation

Tento repozitár obsahuje zdrojový kód k bakalárskej práci **Využitie difúznych modelov pre segmentáciu koronálnych dier**.

Projekt implementuje difúzny model pre binárnu segmentáciu koronálnych dier na snímkach SDO/AIA 193 Å. Model využíva U-Net architektúru v difúznom procese a generuje segmentačné masky pomocou postupného odšumovania.

## Obsah repozitára

Repozitár obsahuje iba zdrojový kód, konfiguračné súbory a pomocné skripty.

Veľké dátové súbory, PNG obrázky, natrénované checkpointy modelov a vygenerované výsledky nie sú súčasťou GitHub repozitára z dôvodu veľkého objemu.

## Inštalácia

Odporúčané je vytvoriť virtuálne prostredie:

    python -m venv .venv

Aktivácia vo Windows PowerShell:

    .\.venv\Scripts\activate

Inštalácia potrebných knižníc:

    pip install -r requirements.txt

## Štruktúra projektu

    train.py              - trénovanie modelu
    sample.py             - generovanie segmentačných masiek
    config.py             - konfigurácia hlavného experimentu
    config_clean.py       - konfigurácia doplnkového cleaned experimentu

    data/                 - dátové triedy a načítanie dát
    models/               - implementácia U-Net architektúry
    diffusion/            - difúzny proces, DDPM/DDIM samplovanie
    utils/                - metriky, seed, EMA a pomocné funkcie
    eval_reiss/           - skripty pre Reiss benchmark
    scripts/              - pomocné skripty pre prípravu dát

## Dáta

Hlavný experiment používa dáta v štýle SCSS-Net:

    data/193_train/
    data/193_test/

Ak hlavné dáta nie sú dostupné, je možné ich stiahnuť pomocou skriptu:

    python scripts/download_data.py

Po stiahnutí alebo manuálnom doplnení dát má byť štruktúra:

    data/193_train/
    data/193_test/

Doplnkový cleaned experiment používa dáta:

    data/cleaned_256/

Pre cleaned experiment musí byť dostupný súbor:

    data/cleaned_256/manifest.csv

Reiss benchmark používa vstupy v priečinku:

    comparison_29/

Tieto veľké dátové priečinky nie sú súčasťou GitHub repozitára a musia byť doplnené samostatne.

## Checkpointy modelov

Natrénované checkpointy nie sú súčasťou GitHub repozitára z dôvodu veľkého objemu.

Ak sú checkpointy dostupné, majú byť uložené takto:

    checkpoints/best.pt
    checkpoints_clean/best.pt

Ak checkpointy nie sú dostupné, model je možné znovu natrénovať pomocou skriptov uvedených nižšie.

## Trénovanie modelu

Hlavný experiment sa spustí príkazom:

    python train.py --mode old

Tento režim používa konfiguráciu:

    config.py

Doplnkový cleaned experiment sa spustí príkazom:

    python train.py --mode clean

Tento režim používa konfiguráciu:

    config_clean.py

Rýchla kontrola, či sa tréning spustí:

    python train.py --mode old --epochs 1

alebo:

    python train.py --mode clean --epochs 1

## Inferencia

Na generovanie segmentačných masiek sa používa:

    python sample.py

Pre inferenciu musí byť dostupný natrénovaný checkpoint modelu.

Finálne nastavenie hlavného modelu použité v práci:

    DDIM steps: 100
    n_samples: 6
    threshold: 0.72
    TTA: flip4
    kernel size: 5
    min area: 200 px

## Reiss benchmark

Spustenie Reiss pipeline pre hlavný model:

    python eval_reiss/run_reiss_pipeline.py --source comparison_29 --ckpt checkpoints/best.pt --work-dir reiss_run --method-name Diffusion-BC --ddim-steps 100 --n-samples 6 --thr 0.68 --tta flip4 --kernel-size 5 --min-area 120 --overwrite

Spustenie Reiss pipeline pre cleaned model:

    python eval_reiss/run_reiss_pipeline.py --source comparison_29 --ckpt checkpoints_clean/best.pt --work-dir reiss_run_clean --method-name Diffusion-BC-Clean --ddim-steps 100 --n-samples 6 --thr 0.68 --tta flip4 --kernel-size 5 --min-area 120 --overwrite

Pre spustenie Reiss benchmarku musia byť dostupné:

    comparison_29/
    checkpoints/best.pt

alebo pre cleaned model:

    checkpoints_clean/best.pt

## Kontrola projektu

Kontrola syntaxe všetkých Python súborov:

    python -m compileall .

Kontrola dostupných argumentov:

    python train.py --help
    python sample.py --help
    python eval_reiss/run_reiss_pipeline.py --help

Kontrola Reiss vstupov:

    python eval_reiss/prepare_manifest.py --source comparison_29 --out-dir tmp_reiss_dataset --overwrite

Očakávaný výsledok:

    Matched cases: 29
    Unmatched inputs: 0
    Unused annotations: 0

## Výsledky použité v práci

Výsledky hlavného modelu na štandardnej testovacej množine:

    Dice:      0.8597
    IoU:       0.7624
    Precision: 0.8510
    Recall:    0.8904

Výsledky hlavného modelu na benchmarku Reiss:

    CH TPR:       0.767
    Filament FPR: 0.096
    Other:        7

Výsledky cleaned modelu na benchmarku Reiss:

    CH_TPR:  0.4767
    Fil_FPR: 0.0000
    Other:   0

## Poznámka

Model je citlivý na formát vstupných dát. Pre správne výsledky musia byť vstupné snímky predspracované rovnakým spôsobom ako pri trénovaní.

GitHub repozitár obsahuje iba kód projektu. Dáta, checkpointy a vygenerované výsledky sa z dôvodu veľkého objemu odovzdávajú samostatne alebo sa model môže znovu natrénovať podľa uvedených príkazov.
