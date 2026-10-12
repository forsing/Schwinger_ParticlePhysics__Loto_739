#!/usr/bin/env python3

"""Schwinger V3 particle physics: spinorski kanali i rang celih prelaza."""

import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

DEFAULTS = (
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv",
)

WINDOW, STARTS = 96, 12
ALPHAS = (.01, .1, 1.)
PAIRS = tuple(
    (i, j) for i in range(4) for j in range(i+1, 4)
)


def load(path):
    rows = []

    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue

            try:
                numbers = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (
                len(numbers) != 7
                or numbers != sorted(numbers)
                or len(set(numbers)) != 7
                or not all(1 <= n <= 39 for n in numbers)
            ):
                raise ValueError(
                    f"Red {line}: neispravna sortirana "
                    "kombinacija 7/39."
                )

            rows.append(numbers)

    return np.asarray(rows, dtype=int).reshape(-1, 7)


def support(history):
    edges = np.column_stack((
        np.zeros(len(history)),
        history,
        np.full(len(history), 40)
    ))
    layout = np.diff(edges, axis=1)/5.

    tree = cKDTree(layout)
    nearest = tree.query(
        layout, k=2, p=1
    )[0][:, 1]

    radius = float(np.quantile(nearest, .75))
    return tree, max(radius, 1e-12)


def supported(rows, tree, radius):
    rows = np.atleast_2d(rows)

    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), 40)
    ))
    distance = tree.query(
        np.diff(edges, axis=1)/5.,
        k=1,
        p=1
    )[0]

    return distance <= radius+1e-10


def spinors(rows):
    rows = np.atleast_2d(rows)

    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), 40)
    ))
    g = np.diff(edges, axis=1)
    middle = (
        edges[:, :-1]+edges[:, 1:]
    )/2.

    psi = (
        np.sqrt(g/40.)
        * np.exp(2j*np.pi*middle/40.)
    )

    return psi.reshape(-1, 4, 2)


def currents(z):
    a, b = z[:, :, 0], z[:, :, 1]
    aa, bb = np.abs(a)**2, np.abs(b)**2
    product = a.conj()*b

    return np.stack((
        aa+bb,
        product.real,
        product.imag,
        .5*(aa-bb)
    ), axis=2)


def state(rows):
    return (
        4.*currents(spinors(rows)).reshape(-1, 16)
    )


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (224, 1))

    for j in range(7):
        result[j*32:(j+1)*32, j] = unused

    return np.sort(result, axis=1)


def features(states, reference, previous_change):
    delta = np.atleast_2d(states)-reference
    vector = delta.reshape(-1, 4, 4)[:, :, 1:]

    coupling = np.column_stack([
        np.sum(vector[:, i]*vector[:, j], axis=1)
        for i, j in PAIRS
    ])

    # Direktan skor prelaza.
    # Nema izlaznog proseka brojeva ili amplituda.
    return np.column_stack((
        delta,
        delta*delta,
        coupling,
        delta*previous_change
    ))


def prepare(history):
    if len(history) < WINDOW+2:
        raise ValueError(
            f"Potrebno je najmanje "
            f"{WINDOW+2} prethodnih redova."
        )

    history = history[-(WINDOW+2):]
    s = state(history)

    gram = np.zeros((54, 54))
    rhs = np.zeros(54)

    for t in range(2, len(history)):
        previous = s[t-1]-s[t-2]

        actual = features(
            s[t:t+1], s[t-1], previous
        )
        other = features(
            state(replacements(history[t])),
            s[t-1],
            previous
        )

        contrast = actual-other
        gram += (
            contrast.T @ contrast/len(contrast)
        )
        rhs += contrast.sum(axis=0)/len(contrast)

    gram /= WINDOW
    rhs /= WINDOW

    scale = np.maximum(
        np.sqrt(np.diag(gram)), 1e-8
    )

    return (
        gram/scale[:, None]/scale[None, :],
        rhs/scale,
        scale,
        s[-1],
        s[-1]-s[-2],
        history[2:]
    )


def fit(prepared, alpha):
    gram, rhs, scale = prepared[:3]

    coefficient = np.linalg.solve(
        gram+alpha*np.eye(54),
        rhs
    )/scale

    return (
        coefficient,
        prepared[3],
        prepared[4]
    )


def scores(rows, model):
    coefficient, reference, previous = model

    result = features(
        state(rows), reference, previous
    ) @ coefficient

    if not np.all(np.isfinite(result)):
        raise ArithmeticError(
            "Neispravan skor spinorskih kanala."
        )

    return result


def search(history, model):
    seeds = np.unique(history, axis=0)
    order = np.argsort(
        -scores(seeds, model),
        kind="stable"
    )
    current = seeds[order[:STARTS]]

    tree, radius = support(history)
    iterations = 0

    while True:
        iterations += 1

        options = np.vstack([
            np.vstack((r, replacements(r)))
            for r in current
        ])

        raw = scores(options, model)
        raw[~supported(options, tree, radius)] = -np.inf
        values = raw.reshape(len(current), 225)

        chosen = np.argmax(values, axis=1)
        better = (
            values[np.arange(len(current)), chosen]
            > values[:, 0]+1e-10
        )

        if not np.any(better):
            best = int(np.argmax(values[:, 0]))

            return (
                tuple(map(int, current[best])),
                float(values[best, 0]),
                iterations
            )

        blocks = options.reshape(
            len(current), 225, 7
        )
        current[better] = blocks[
            np.arange(len(current))[better],
            chosen[better]
        ]


def predict(prepared, alpha):
    return search(
        prepared[5],
        fit(prepared, alpha)
    )


def hits(prediction, actual):
    return len(
        set(map(int, prediction))
        & set(map(int, actual))
    )


def summarize(label, values):
    print(
        f"{label}: prosek={np.mean(values):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(values, minlength=8).tolist()}",
        flush=True
    )


def run_file(path, validation_count=48, test_count=64):
    rows = load(path)
    split = len(rows)-test_count
    begin = split-validation_count

    if (
        validation_count < 16
        or test_count < 16
        or begin < WINDOW+2
    ):
        raise ValueError(
            "Premalo istorije ili manje od "
            "16 validacionih/test kola."
        )

    print(
        f"\nUlaz: {Path(path).resolve()}; "
        f"redova={len(rows)}",
        flush=True
    )

    validation = [[] for _ in ALPHAS]

    for t in range(begin, split):
        p = prepare(rows[:t])

        for j, alpha in enumerate(ALPHAS):
            combo, _, _ = predict(p, alpha)
            validation[j].append(
                hits(combo, rows[t])
            )

        if (t-begin+1) % 16 == 0:
            print(
                f"Validacija: "
                f"{t-begin+1}/{validation_count}",
                flush=True
            )

    means = [np.mean(v) for v in validation]
    alpha = ALPHAS[int(np.argmax(means))]

    print(
        f"Validacija: alpha={alpha:g}; "
        f"prosek={max(means):.4f}",
        flush=True
    )

    model_scores, previous_scores = [], []

    # Alpha ostaje fiksna tokom testa.
    # Model i empirijska podrska koriste samo prethodne redove.
    for t in range(split, len(rows)):
        combo, _, _ = predict(
            prepare(rows[:t]), alpha
        )

        model_scores.append(
            hits(combo, rows[t])
        )
        previous_scores.append(
            hits(rows[t-1], rows[t])
        )

        if (t-split+1) % 16 == 0:
            print(
                f"Test: {t-split+1}/{test_count}",
                flush=True
            )

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize(
        "SCHWINGER V3 PARTICLE PHYSICS",
        model_scores
    )
    summarize(
        "Prethodna kombinacija",
        previous_scores
    )

    p = prepare(rows)
    combo, score, iterations = predict(p, alpha)

    repeated = any(
        np.array_equal(combo, row)
        for row in rows
    )

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Skor={score:.10f}; "
        f"iteracija={iterations}; "
        f"vec postoji u istoriji={repeated}"
    )

    tree, radius = support(p[5])
    print(
        "Empirijska podrska: "
        f"L1 poluprecnik={radius:.6f}; kvantil=.75"
    )
    print(
        "Proveren lokalni maksimum dozvoljenih zamena; "
        "globalni maksimum nije potvrden."
    )
    print(
        "Spinorski kanali su strukturna analogija; "
        "skor nije verovatnoca dobitka."
    )


def self_test():
    base = np.array([
        [1,2,3,4,5,6,7],
        [4,12,18,28,35,36,37],
        [33,34,35,36,37,38,39],
        [2,7,12,18,24,30,39]
    ])

    z = spinors(base)
    c = currents(z)

    assert np.allclose(
        np.sum(c[:, :, 1:]**2, axis=2),
        .25*c[:, :, 0]**2
    )

    density = np.stack((
        .5*c[:, :, 0]+c[:, :, 3],
        .5*c[:, :, 0]-c[:, :, 3]
    ), axis=2).reshape(-1, 8)

    assert np.allclose(
        40*np.cumsum(
            density[:, :-1], axis=1
        ),
        base
    )

    rotation = np.exp(
        1j*np.array([.2, .7, 1.1, -.4])
    )[None, :, None]
    assert np.allclose(
        currents(z*rotation), c
    )

    u = np.array([
        [np.cos(.3), -np.sin(.3)],
        [np.sin(.3), np.cos(.3)]
    ])
    rotated = currents(z @ u.T)

    assert np.allclose(
        rotated[:, :, 0], c[:, :, 0]
    )
    assert np.allclose(
        np.sum(rotated[:, :, 1:]**2, axis=2),
        np.sum(c[:, :, 1:]**2, axis=2)
    )

    history = np.tile(base, (30, 1))
    p = prepare(history)

    for alpha in ALPHAS:
        model = fit(p, alpha)
        coef = model[0]*p[2]

        assert np.allclose(
            (p[0]+alpha*np.eye(54)) @ coef,
            p[1]
        )

        a = predict(p, alpha)
        assert a == predict(
            prepare(history), alpha
        )

        tree, radius = support(p[5])
        r = replacements(a[0])
        allowed = supported(r, tree, radius)

        assert (
            scores(r[allowed], model).max(
                initial=-np.inf
            )
            <= a[1]+1e-9
        )
        assert supported(
            a[0], tree, radius
        )[0]

    changed = history.copy()
    changed[-1] = base[0]

    assert (
        predict(prepare(history[:-1]), .1)
        == predict(prepare(changed[:-1]), .1)
    )

    print(
        "Self-test OK: spinori, identiteti, "
        "rekonstrukcija, ucenje i hronologija."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*")
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    for path in args.csv or DEFAULTS:
        run_file(path, args.validation, args.test)


if __name__ == "__main__":
    main()



"""
Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: alpha=0.1; prosek=1.4792
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
SCHWINGER V3 PARTICLE PHYSICS: prosek=1.1094; pogodaka 0..7=[16, 27, 19, 2, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 1,2,3,4,6,12,13
Skor=1.1613503262; iteracija=28; vec postoji u istoriji=False
Empirijska podrska: L1 poluprecnik=2.800000; kvantil=.75
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
Spinorski kanali su strukturna analogija; skor nije verovatnoca dobitka.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: alpha=0.1; prosek=1.2708
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
SCHWINGER V3 PARTICLE PHYSICS: prosek=1.4375; pogodaka 0..7=[13, 20, 22, 8, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 1,2,7,9,15,38,39
Skor=1.5155870456; iteracija=15; vec postoji u istoriji=False
Empirijska podrska: L1 poluprecnik=2.800000; kvantil=.75
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
Spinorski kanali su strukturna analogija; skor nije verovatnoca dobitka.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: alpha=1; prosek=1.3958
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
SCHWINGER V3 PARTICLE PHYSICS: prosek=1.3125; pogodaka 0..7=[9, 31, 20, 3, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 1,2,20,36,37,38,39
Skor=0.8959685867; iteracija=16; vec postoji u istoriji=False
Empirijska podrska: L1 poluprecnik=3.200000; kvantil=.75
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
Spinorski kanali su strukturna analogija; skor nije verovatnoca dobitka.
"""



"""
V3 koristi Schwingerovu ideju parova oscilatora za opis spina, prenetu na četiri para razmaka. Uči skor celog prelaza i spregu između tih kanala. Provere rekonstrukcije kombinacije, fazne simetrije i svih pojedinačnih zamena su prošle.

Provera je pokazala da slobodna pretraga može otići do ekstremno zbijenog rasporeda, daleko od podataka. Ograničiću je oblašću koju određuju stvarni razmaci iz istorije. 

Nova ideja su spinorski kanali i njihove sprege, inspirisani Schwingerovim opisom ugaonog momenta parovima oscilatora. 
Model rangira cele prelaze. Pretraga ostaje u oblasti određenoj stvarnim istorijskim razmacima, bez nametnute raspodele ili aritmetičkih početnih nizova.

Fajl	                Sledeća kombinacija	    Prosek pogodaka na 64 kola	Prethodna kombinacija
Objedinjeni, red 4699	1,2,3,4,6,12,13	        1,1094	                    1,2188
Loto, red 2972	        1,2,7,9,15,38,39	    1,4375	                    1,2969
Loto Plus, red 1728	    1,2,20,36,37,38,39	    1,3125	                    1,2969

Provereni su rekonstrukcija, fazna simetrija, hronologija, ponovljivost i lokalni maksimum dozvoljenih zamena. Nijedan predlog već ne postoji u svom CSV-u.

"""



"""
Julian Seymour Schwinger  
1-quantum electrodynamics.   2-nuclear physics.     3-particle physics.     
4-atomic physics       5-quantum field theory.        6-Schwinger effect
"""
