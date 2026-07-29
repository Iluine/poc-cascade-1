# pocCascade2phys — Test T1

Routage **deux-physiques** sur interface fluide-structure (FSI) contestée.
PoC pré-enregistré ; voir [`PREREGISTRATION.md`](PREREGISTRATION.md) — **le contrat fait foi**.

> **STATUT (2026-06-28) : T1 CLOS.** Instrument FSI validé (G0 a/b/c) + pipeline complet et testé.
> **Portée exacte du verdict : la décision routeur s'est effondrée en 1 sortie sur 3** (mémoïser
> 100 %, expert 0 %, descend 0 %). T1 a mesuré le domaine de validité de la sortie **cache/expert**
> (perceptuellement vacant à Re=100 2D — non-récurrence < JND) ; la sortie **descend, cœur de la
> thèse, reste NON TESTÉE** (substrat tout-bas-rang, jamais sollicitée). Verdict honnête mais sur
> **2 sorties sur 3**. Pas de verdict fabriqué. Suite = **T1.5 = substrat sollicitant les TROIS
> sorties** (le chaos Re-élevé est peut-être nécessaire pour réveiller le descend ; l'apériodique
> à Re=100 ne réveille que l'expert). Détails : conclusion de [`PREREGISTRATION.md`](PREREGISTRATION.md).

## Ce que T1 tranche (un seul claim)

Un routeur évalué **par fenêtre de Harten, sur le seul halo grossier**, décide entre
(1) réutiliser un opérateur mémoïsé, (2) invoquer un expert de régime, (3) descendre
d'un niveau — pour que le `F` composite routé reproduise **en espace perceptuel** un
écoulement FSI couplé (sillage **et** interface), sur **horizon long**, à **compute
matériellement inférieur** au surrogate fin partout, et **mieux qu'un routeur de
Harten codé en dur**.

> **G0/G0' valident l'INSTRUMENT. C1–C4 valident l'ARCHITECTURE.** Jamais l'inverse.

## Stack (réalité au 2026-06-26)

| Rôle | Brique | Note |
|---|---|---|
| Oracle fluide | **XLB** (Lattice-Boltzmann diff.) | repli de AegirJAX (inexistant) — §1 |
| Oracle solide | **jax-fem** (élasticité FEM) | v0.0.12 |
| Couplage | **Frontière immergée** (Peskin), JAX | monolithique, sans remaillage |
| GPU | RTX 3050 Ti 4 Go, CUDA 12.8 | `jax[cuda12]==0.10.2` |

## Architecture construite (la contribution)

```
cascade/
  fluid/       ✅ wrapper XLB        — oracle fluide + champ d'entrée des experts
  solid/       ⬜ VIDE (planifié)    — wrapper jax-fem (config poteau, T1.5)
  coupling/    ✅ IB Peskin          — bus minimal = (ρ, ρu) (RECADRAGE gravé)
  harten/      ✅ fenêtrage 2–3 niv. — granularité de routage native
  router/      ⬜ VIDE (planifié)    — le routeur nul vit inline dans les expériences C2
  experts/     ✅ POD+DMD fluide     — régimes physiques (anti-circularité)
  metrics/     ✅ M1–M4 perceptuel   — JAMAIS L2 comme critère
  accounting/  ⬜ VIDE (planifié)    — requis pour C3
  configs/     ⬜ VIDE (planifié)
```
> Marquage ✅/⬜ ajouté le 2026-07-29 : ce README présentait les 9 modules comme
> construits ; 5 le sont (cf. CLAUDE.md, même table). C2 est perceptuellement
> VACANT à Re=100 2D — T1 clos, suite = T1.5 (`experiments/mixed_substrate.py`).

## Ordre d'exécution (gaté)

1. **F0–F3** fondation : env GPU, XLB+jax-fem, couplage IB, Harten + métriques + accounting.
2. **Config ressort** : **G0** (gate instrument) → **C1** → **Phase 1** → **C2/C3/C4**
   sur ≥ 100 `T_shed`, ≥ 5 seeds.
3. Si la boucle tourne : **config poteau**, **G0'**, répéter C1→C4 (discriminateur).
4. **Phase 2** différentiable seulement si Phase 1 positive.
5. Trancher « meilleur » au sens §11.

> **Le premier chiffre qui compte sort à G0 (config ressort).**

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv "jax[cuda12]==0.10.2"
# XLB + jax-fem : voir F1
```
