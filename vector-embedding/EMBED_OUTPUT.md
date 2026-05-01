# python embed.py 

============================================================

  Vector Embedding Demo  —  all-MiniLM-L6-v2

============================================================

## [1/4] Loading model (all-MiniLM-L6-v2) …

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 13886.64it/s]

## [2/4] Generating embeddings …

  Embedding dimension : 384
  Words               : ['king', 'queen', 'man', 'woman', 'prince', 'princess', 'boy', 'girl', 'lord', 'lady']

## [3/4] Pairwise cosine similarity  (1 = identical, 0 = orthogonal, -1 = opposite)

------------------------------------------------------------

```text
╭─────────────────────┬──────────────┬─────────────────┬──────────────────╮
│ Pair                │   Cosine Sim │ Visual          │ Relationship     │
├─────────────────────┼──────────────┼─────────────────┼──────────────────┤
│ king  ↔  queen      │       0.6807 │ █████████████   │ somewhat related │
│ king  ↔  man        │       0.3216 │ ██████          │ weakly related   │
│ king  ↔  woman      │       0.264  │ █████           │ weakly related   │
│ king  ↔  prince     │       0.5884 │ ███████████     │ somewhat related │
│ king  ↔  princess   │       0.4843 │ █████████       │ somewhat related │
│ king  ↔  boy        │       0.3833 │ ███████         │ weakly related   │
│ king  ↔  girl       │       0.2726 │ █████           │ weakly related   │
│ king  ↔  lord       │       0.4215 │ ████████        │ somewhat related │
│ king  ↔  lady       │       0.3647 │ ███████         │ weakly related   │
│ queen  ↔  man       │       0.2541 │ █████           │ weakly related   │
│ queen  ↔  woman     │       0.4394 │ ████████        │ somewhat related │
│ queen  ↔  prince    │       0.5789 │ ███████████     │ somewhat related │
│ queen  ↔  princess  │       0.6044 │ ████████████    │ somewhat related │
│ queen  ↔  boy       │       0.2769 │ █████           │ weakly related   │
│ queen  ↔  girl      │       0.3954 │ ███████         │ weakly related   │
│ queen  ↔  lord      │       0.4255 │ ████████        │ somewhat related │
│ queen  ↔  lady      │       0.5636 │ ███████████     │ somewhat related │
│ man  ↔  woman       │       0.3257 │ ██████          │ weakly related   │
│ man  ↔  prince      │       0.2717 │ █████           │ weakly related   │
│ man  ↔  princess    │       0.2265 │ ████            │ weakly related   │
│ man  ↔  boy         │       0.5843 │ ███████████     │ somewhat related │
│ man  ↔  girl        │       0.4362 │ ████████        │ somewhat related │
│ man  ↔  lord        │       0.5849 │ ███████████     │ somewhat related │
│ man  ↔  lady        │       0.4127 │ ████████        │ somewhat related │
│ woman  ↔  prince    │       0.2634 │ █████           │ weakly related   │
│ woman  ↔  princess  │       0.4024 │ ████████        │ somewhat related │
│ woman  ↔  boy       │       0.258  │ █████           │ weakly related   │
│ woman  ↔  girl      │       0.5063 │ ██████████      │ somewhat related │
│ woman  ↔  lord      │       0.2511 │ █████           │ weakly related   │
│ woman  ↔  lady      │       0.7727 │ ███████████████ │ similar          │
│ prince  ↔  princess │       0.6806 │ █████████████   │ somewhat related │
│ prince  ↔  boy      │       0.3058 │ ██████          │ weakly related   │
│ prince  ↔  girl     │       0.2983 │ █████           │ weakly related   │
│ prince  ↔  lord     │       0.382  │ ███████         │ weakly related   │
│ prince  ↔  lady     │       0.3548 │ ███████         │ weakly related   │
│ princess  ↔  boy    │       0.3388 │ ██████          │ weakly related   │
│ princess  ↔  girl   │       0.4691 │ █████████       │ somewhat related │
│ princess  ↔  lord   │       0.3669 │ ███████         │ weakly related   │
│ princess  ↔  lady   │       0.5012 │ ██████████      │ somewhat related │
│ boy  ↔  girl        │       0.6162 │ ████████████    │ somewhat related │
│ boy  ↔  lord        │       0.4566 │ █████████       │ somewhat related │
│ boy  ↔  lady        │       0.3721 │ ███████         │ weakly related   │
│ girl  ↔  lord       │       0.4111 │ ████████        │ somewhat related │
│ girl  ↔  lady       │       0.5829 │ ███████████     │ somewhat related │
│ lord  ↔  lady       │       0.4666 │ █████████       │ somewhat related │
╰─────────────────────┴──────────────┴─────────────────┴──────────────────╯
```

## [4/4] Pairwise Euclidean distance  (lower = closer)

------------------------------------------------------------

```text
╭─────────────────────┬──────────────────────╮
│ Pair                │   Euclidean Distance │
├─────────────────────┼──────────────────────┤
│ woman  ↔  lady      │               0.6742 │
│ king  ↔  queen      │               0.7991 │
│ prince  ↔  princess │               0.7992 │
│ boy  ↔  girl        │               0.8761 │
│ queen  ↔  princess  │               0.8895 │
│ king  ↔  prince     │               0.9073 │
│ man  ↔  lord        │               0.9112 │
│ man  ↔  boy         │               0.9118 │
│ girl  ↔  lady       │               0.9133 │
│ queen  ↔  prince    │               0.9178 │
│ queen  ↔  lady      │               0.9342 │
│ woman  ↔  girl      │               0.9937 │
│ princess  ↔  lady   │               0.9988 │
│ king  ↔  princess   │               1.0156 │
│ princess  ↔  girl   │               1.0304 │
│ lord  ↔  lady       │               1.0328 │
│ boy  ↔  lord        │               1.0425 │
│ queen  ↔  woman     │               1.0589 │
│ man  ↔  girl        │               1.0619 │
│ queen  ↔  lord      │               1.0719 │
│ king  ↔  lord       │               1.0757 │
│ man  ↔  lady        │               1.0838 │
│ girl  ↔  lord       │               1.0853 │
│ woman  ↔  princess  │               1.0933 │
│ queen  ↔  girl      │               1.0997 │
│ king  ↔  boy        │               1.1106 │
│ prince  ↔  lord     │               1.1118 │
│ boy  ↔  lady        │               1.1206 │
│ princess  ↔  lord   │               1.1252 │
│ king  ↔  lady       │               1.1272 │
│ prince  ↔  lady     │               1.1359 │
│ princess  ↔  boy    │               1.15   │
│ man  ↔  woman       │               1.1613 │
│ king  ↔  man        │               1.1648 │
│ prince  ↔  boy      │               1.1783 │
│ prince  ↔  girl     │               1.1847 │
│ queen  ↔  boy       │               1.2026 │
│ king  ↔  girl       │               1.2061 │
│ man  ↔  prince      │               1.2069 │
│ king  ↔  woman      │               1.2133 │
│ woman  ↔  prince    │               1.2138 │
│ woman  ↔  boy       │               1.2182 │
│ queen  ↔  man       │               1.2214 │
│ woman  ↔  lord      │               1.2238 │
│ man  ↔  princess    │               1.2438 │
╰─────────────────────┴──────────────────────╯
```

============================================================

  Analogy:  king  −  man  +  woman  =  ?
  
============================================================

  Nearest word  →  queen  (similarity=0.5795)

  Similarity of (king − man + woman) to each word:

```text
╭──────────┬──────────────╮
│ Word     │   Similarity │
├──────────┼──────────────┤
│ king     │       0.6306 │
│ woman    │       0.6279 │
│ queen    │       0.5795 │
│ lady     │       0.4849 │
│ princess │       0.4418 │
│ prince   │       0.3882 │
│ girl     │       0.2294 │
│ lord     │       0.0587 │
│ boy      │       0.0381 │
│ man      │      -0.236  │
╰──────────┴──────────────╯
```

  Interpretation: The resulting vector is closest to 'queen',
  confirming that these embeddings capture gender & royalty semantics.