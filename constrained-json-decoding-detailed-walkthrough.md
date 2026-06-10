# How vLLM Guarantees Valid JSON: A Step-by-Step Walkthrough of Constrained Decoding

A relentlessly concrete trace of how a JSON Schema becomes a finite-state machine, how that machine masks the model's logits at every decode step, and why the output is **guaranteed** to parse — worked through the purchase-order schema, token by token.

---

## Table of Contents

1. [The Problem: Sampling Doesn't Know About JSON](#1-the-problem)
2. [The Big Picture: Three Stages](#2-the-big-picture)
3. [The Toy System: Vocabulary and Schema](#3-the-toy-system)
4. [Stage 1 — Schema → Regular Expression](#4-stage-1-schema-to-regex)
5. [Stage 2 — Regex → Character DFA](#5-stage-2-regex-to-dfa)
6. [Stage 3 — The Index: a Token→State Transition Map](#6-stage-3-the-index)
7. [Stage 4 — The Runtime Decode Loop (masking the logits)](#7-stage-4-the-runtime-loop)
8. [The Full Token-by-Token Trace](#8-the-full-trace)
9. [Zoom In: How the Enum Collapses](#9-enum-collapse)
10. [Zoom In: Tokenizer Alignment and Boundary-Crossing Tokens](#10-tokenizer-alignment)
11. [Why It Is Guaranteed (the invariant)](#11-the-guarantee)
12. [The Real Libraries: Outlines, XGrammar, lm-format-enforcer](#12-real-libraries)
13. [Scale: Toy vs Production](#13-scale)
14. [Limitations and Gotchas](#14-limitations)
15. [Summary](#15-summary)

---

<a name="1-the-problem"></a>
## 1. The Problem: Sampling Doesn't Know About JSON

An LLM generates one token at a time. At each step it produces a vector of **logits** — one real number per vocabulary token — turns them into probabilities with softmax, and samples one token.

```
logits  (|V| numbers)  ──softmax──▶  probabilities  ──sample──▶  one token
```

Nothing in this process knows that you wanted JSON. If the model is decoding the value of `"quantity"` and the highest-probability token happens to be `"` or the word `about`, it will emit it — and your `json.loads()` throws. Prompt-engineering ("respond ONLY with valid JSON") lowers the probability of mistakes but **never makes it zero**.

Constrained decoding makes it exactly zero. The idea in one sentence:

> Before sampling, look at which tokens are *legal* given everything generated so far, and set the logits of every *illegal* token to −∞. After softmax, illegal tokens have probability 0. The model literally cannot pick them.

The entire engineering problem is answering, fast enough to run every single step, the question: **"given the partial output so far, which tokens are legal?"** That answer comes from a finite-state machine compiled from your schema.

---

<a name="2-the-big-picture"></a>
## 2. The Big Picture: Three Stages

Two stages happen **once**, when you register the schema (compile time). One stage happens **every decode step** (run time).

```
   COMPILE TIME (once per schema)                 RUN TIME (every token)
   ┌───────────────────────────┐                 ┌────────────────────────┐
   │ JSON Schema                │                 │  model produces logits │
   │      │                     │                 │         │              │
   │      ▼  Stage 1            │                 │         ▼              │
   │ Regular expression         │                 │  look up allowed-token │
   │      │                     │                 │  set for current state │
   │      ▼  Stage 2            │                 │         │              │
   │ Character DFA (states)     │                 │         ▼              │
   │      │                     │                 │  mask illegal → −∞     │
   │      ▼  Stage 3            │                 │         │              │
   │ Index: state → {token →    │ ───────────────▶│         ▼              │
   │        next state}         │   (used here)   │  softmax + sample      │
   └───────────────────────────┘                 │         │              │
                                                  │         ▼              │
                                                  │  advance DFA state     │
                                                  └────────────────────────┘
```

The crucial trick (the contribution of the Outlines paper, Willard & Louf 2023) is **Stage 3**: precompute, for every FSM state, exactly which *vocabulary tokens* are allowed and where each one lands you. Then run time is just a dictionary lookup plus a masked softmax — microseconds, not a re-parse of the whole string.

---

<a name="3-the-toy-system"></a>
## 3. The Toy System: Vocabulary and Schema

### 3.1 The schema (our purchase order)

```python
po_schema = {
  "type": "object",
  "properties": {
    "supplier":      {"type": "string"},
    "item":          {"type": "string"},
    "quantity":      {"type": "integer", "minimum": 1},
    "currency":      {"enum": ["USD", "EUR", "INR"]},
    "delivery_date": {"type": "string", "format": "date"}
  },
  "required": ["supplier", "item", "quantity"],
  "additionalProperties": False
}
```

### 3.2 The toy tokenizer

A real tokenizer has 50k–130k tokens. Ours has **31**, chosen to expose every interesting case: merged punctuation tokens, multi-character content tokens, digit tokens, and enum fragments.

| ID | Token (the literal characters) | Kind |
|---:|---|---|
| 0 | `<EOS>` | end-of-sequence |
| 1 | `{` | structural |
| 2 | `}` | structural |
| 3 | `"` | structural |
| 4 | `:` | structural |
| 5 | `,` | structural |
| 6 | `{"` | **merged** (brace + quote) |
| 7 | `":"` | **merged** (quote + colon + quote) |
| 8 | `","` | **merged** (quote + comma + quote) |
| 9 | `supplier` | key literal |
| 10 | `item` | key literal |
| 11 | `quantity` | key literal |
| 12 | `currency` | key literal |
| 13 | `delivery_date` | key literal |
| 14 | `Acme` | string content |
| 15 | `bolt` | string content |
| 16 | `bolts` | string content |
| 17 | `USD` | enum value |
| 18 | `EUR` | enum value |
| 19 | `INR` | enum value |
| 20 | `US` | enum fragment |
| 21 | `D` | letter |
| 22 | `2` | digit |
| 23 | `20` | digit run |
| 24 | `200` | digit run |
| 25 | `0` | digit |
| 26 | `1` | digit |
| 27 | `5` | digit |
| 28 | `x` | letter |
| 29 | `":` | **merged** (quote + colon) |
| 30 | `,"` | **merged** (comma + quote) |

> A token is just a string of characters with an ID. Notice IDs 6, 7, 8, 29, 30 each span **multiple** grammar characters, and some even **cross a structural boundary** (e.g. `","` is "close one string, comma, open the next"). Real BPE tokenizers are full of these. Handling them correctly is the whole game in Section 10.

---

<a name="4-stage-1-schema-to-regex"></a>
## 4. Stage 1 — Schema → Regular Expression

The compiler walks the schema and emits a regex that matches **exactly** the JSON strings the schema permits. Each schema construct has a regex fragment:

| Schema construct | Regex fragment | Why |
|---|---|---|
| `{"type":"string"}` | `"[^"]*"` | a quote, any non-quote chars, a quote |
| `{"type":"integer","minimum":1}` | `[1-9][0-9]*` | first digit 1–9 (no leading zero, ≥ 1), then any digits |
| `{"enum":["USD","EUR","INR"]}` | `"(USD\|EUR\|INR)"` | an alternation of the literal choices |
| `{"type":"string","format":"date"}` | `"[0-9]{4}-[0-9]{2}-[0-9]{2}"` | the date pattern |
| required keys, fixed order | literal `"key":` chained with `,` | structure |
| optional key | wrap its `,"key":value` in `( ... )?` | the `?` makes it skippable |
| `additionalProperties:false` | *no* "any other key" branch is emitted | nothing else can appear |

Composing the whole object (whitespace omitted for clarity; real compilers also allow optional spaces):

```
\{"supplier":"[^"]*","item":"[^"]*","quantity":[1-9][0-9]*(,"currency":"(USD|EUR|INR)")?(,"delivery_date":"[0-9]{4}-[0-9]{2}-[0-9]{2}")?\}
```

Read it left to right:

```
\{                       the opening brace
"supplier":"[^"]*"       required string field
,"item":"[^"]*"          required string field
,"quantity":[1-9][0-9]*  required integer ≥ 1
(,"currency":"(USD|EUR|INR)")?            optional enum
(,"delivery_date":"[0-9]{4}-[0-9]{2}-[0-9]{2}")?   optional date
\}                       the closing brace
```

Two facts already fall out of this regex, before any model runs:

- **`item` is mandatory and ordered.** There is no path from `quantity`'s end back to `supplier`, and no path that skips `item`. So an object like `{"supplier":"Acme","quantity":200}` (the one from the chat, which drops `item`) is **not in the language** — the FSM will refuse to let the model close the brace until `item` has been emitted.
- **`minimum:1` is structural, not a post-check.** It is baked into `[1-9]` for the first digit, so a leading `0` is impossible to emit in the first place.

> Pure regex covers "flat" schemas like this one. Arbitrarily nested / recursive JSON (objects inside arrays inside objects) needs a **context-free grammar + pushdown automaton**, which is what XGrammar uses. The masking principle is identical; only the automaton is more powerful.

---

<a name="5-stage-2-regex-to-dfa"></a>
## 5. Stage 2 — Regex → Character DFA

The regex is compiled (Thompson construction → subset construction) into a **deterministic finite automaton** over *characters*. Every state has at most one transition per character; a missing transition means "dead — this character is illegal here."

Internally there is one state per character position, so the literal `"supplier":"` is a chain of 12 single-character states. To keep the diagram readable, we group the fixed literal runs and show the **macro-states** where real choices happen:

```
            {"          supplier        ":"        [^"]*
 (A0)──────▶(A1)──────▶(KEY1)──────▶(SUP)──────▶ (SUP)        loop on non-quote
 START      need        need          string        │
            "supplier   ":"           body          │ "
                                                     ▼
                                            (close, then ,"item":")
                                                     │
                                                     ▼
 ... item the same way ...                         (ITEM body)
                                                     │ "
                                                     ▼
            ,"quantity":                            (INT0)  need [1-9]
                                                     │ 1-9
                                                     ▼
                                                   (INT1)  need [0-9]*  ── loop 0-9
                                                     │
              ┌──────────────────────┬──────────────┴───────────┐
              │ }                    │ ,"currency":"             │ ,"delivery_date":"
              ▼                      ▼                           ▼
           (ACCEPT)              (ENUM0)                      (DATE body)
                                    │ U|E|I ...                  │
                                    ▼                            ▼
                                 (ENUM trie)                  (after date)
                                    │ "                          │ }
                                    ▼                            ▼
                              (AFTER_CUR)                     (ACCEPT)
                               │ }   │ ,"delivery_date":"
                               ▼     ▼
                           (ACCEPT)(DATE body)
```

The two sub-machines worth drawing in full are the **integer** and the **enum trie**, because that is where the mask gets interesting.

### 5.1 The integer sub-DFA: `[1-9][0-9]*`

```
        1..9                 0..9
 (INT0) ─────▶ (INT1) ◀──────────────┐
   │             │  └─────────────────┘  (loop: stay in INT1 on any 0-9)
   │             │
   │             ├── , ──▶ (next field)
   │             └── } ──▶ (ACCEPT)
   │
   └── 0 ?  NO TRANSITION  →  '0' is illegal as the first digit
```

- From `INT0`, only `1,2,3,4,5,6,7,8,9` have transitions. `0` is a dead character. (This is `minimum:1`.)
- From `INT1`, all of `0..9` are legal (loop), **and** the structural `,` or `}` are legal (the number can end).

### 5.2 The enum sub-DFA: a trie of `USD | EUR | INR`

```
                 U          S          D
 (ENUM0) ──────▶(e_U)─────▶(e_US)────▶(e_USD)──┐
   │  E                                         │
   ├──────▶(e_E)──U──▶(e_EU)──R──▶(e_EUR)──────┤  " 
   │  I                                         ├────▶ (AFTER_CUR)
   └──────▶(e_I)──N──▶(e_IN)──R──▶(e_INR)──────┘
```

From `ENUM0`, the only legal *characters* are `U`, `E`, `I`. Everything else is dead. This trie is why "the enum collapses to only legal continuations" — there is physically no edge for any other letter.

---

<a name="6-stage-3-the-index"></a>
## 6. Stage 3 — The Index: a Token→State Transition Map

The DFA is defined over **characters**, but the model emits **tokens** (multi-character strings). Bridging that gap *at run time* would mean, for every step, trying to feed each token's characters through the DFA — too slow. So we precompute it **once**.

### 6.1 The precompute algorithm

```
for each DFA state s:
    allowed[s] = {}                       # map: token_id -> destination state
    for each token t in vocabulary V:
        state = s
        ok = True
        for each character c in t.string:        # walk the token's chars
            if DFA has a transition state --c--> state2:
                state = state2
            else:
                ok = False; break                # token would derail the FSM
        if ok:
            allowed[s][t.id] = state             # t is legal from s; lands in `state`
```

This produces, for each state, a **dictionary** `{legal token id → next state}`. That dictionary is the "transition map" / "index." Cost is paid once: `O(|states| x |V| x avg_token_length)`. For our toy that is roughly `30 x 31 x 4 ≈ 3,700` character-steps — trivial. (Production numbers in Section 13.)

### 6.2 Worked example: the allowed set for state `ENUM0`

Walk all 31 tokens through the enum trie starting at `ENUM0`:

| Token | Chars walked from ENUM0 | Result | In `allowed[ENUM0]`? |
|---|---|---|---|
| `USD` (17) | U→e_U, S→e_US, D→e_USD | lands in `e_USD` | ✅ → e_USD |
| `EUR` (18) | E→e_E, U→e_EU, R→e_EUR | lands in `e_EUR` | ✅ → e_EUR |
| `INR` (19) | I→e_I, N→e_IN, R→e_INR | lands in `e_INR` | ✅ → e_INR |
| `US` (20) | U→e_U, S→e_US | lands in `e_US` | ✅ → e_US |
| `D` (21) | D → *no edge from ENUM0* | dead | ❌ |
| `Acme` (14) | A → *no edge* | dead | ❌ |
| `200` (24) | 2 → *no edge* | dead | ❌ |
| `"` (3) | `"` → *no edge from ENUM0* | dead | ❌ |
| `{` (1) | `{` → *no edge* | dead | ❌ |
| ...all others... | first char has no edge | dead | ❌ |

```
allowed[ENUM0] = { 17:e_USD, 18:e_EUR, 19:e_INR, 20:e_US }
```

Only **4 of 31** tokens survive. The model's full vocabulary has been pruned to exactly the four ways to start a legal currency. That pruning is the mask.

### 6.3 Worked example: the allowed set for state `INT0`

| Token | First char | Legal from INT0? | Lands in |
|---|---|---|---|
| `1` (26) | 1 | ✅ | INT1 |
| `5` (27) | 5 | ✅ | INT1 |
| `2` (22) | 2 | ✅ | INT1 |
| `20` (23) | 2,0 | ✅ (2 legal, then 0 legal in INT1) | INT1 |
| `200` (24) | 2,0,0 | ✅ | INT1 |
| `0` (25) | 0 | ❌ (no edge for 0 from INT0) | — |
| `Acme` (14) | A | ❌ | — |

```
allowed[INT0] = { 26:INT1, 27:INT1, 22:INT1, 23:INT1, 24:INT1 }
```

Note `200` (a three-character token) is admitted whole from `INT0` because *all three* of its characters trace a legal path. The DFA never sees "200" as a unit — it sees the character stream `2`,`0`,`0` — which is exactly why multi-character tokens just work.

---

<a name="7-stage-4-the-runtime-loop"></a>
## 7. Stage 4 — The Runtime Decode Loop (masking the logits)

Now the model runs. The engine keeps one integer: the **current FSM state**. Each step:

```
1. forward pass            → logits  L  (a vector of length |V| = 31)
2. allowed = index[state]  → the dict of legal token ids for this state
3. build mask m:  m[i] = 0   if i in allowed
                  m[i] = -inf if i not in allowed
4. L' = L + m              → illegal logits become -inf
5. p = softmax(L')         → illegal tokens get probability 0
6. sample token t ~ p
7. state = allowed[t]      → advance the FSM
8. if state == ACCEPT and t == <EOS>: stop
```

### 7.1 A concrete masking computation

Suppose we are at `ENUM0` (about to choose the currency) and the raw model logits over six relevant tokens are:

| Token | `Acme`(14) | `USD`(17) | `US`(20) | `EUR`(18) | `200`(24) | `INR`(19) |
|---|---:|---:|---:|---:|---:|---:|
| raw logit | **4.0** | 3.0 | 2.0 | 1.0 | 2.5 | 0.5 |

Left to itself the model most wants `Acme` (logit 4.0) — a string fragment, totally illegal here. Apply the mask `allowed[ENUM0] = {17,18,19,20}`:

| Token | masked logit | exp(logit) | probability |
|---|---:|---:|---:|
| `Acme`(14) | −inf | 0 | **0.000** |
| `200`(24) | −inf | 0 | **0.000** |
| `USD`(17) | 3.0 | 20.09 | 0.631 |
| `US`(20) | 2.0 | 7.39 | 0.232 |
| `EUR`(18) | 1.0 | 2.72 | 0.085 |
| `INR`(19) | 0.5 | 1.65 | 0.052 |

```
sum of exp over legal tokens = 20.09 + 7.39 + 2.72 + 1.65 = 31.85
P(USD) = 20.09 / 31.85 = 0.631
```

The illegal `Acme`, which had the **largest** raw logit, is now impossible. The probability mass is renormalized across only the legal tokens — we are still sampling from the model's preferences, just restricted to the legal subset. The model picks `USD` and the FSM advances `ENUM0 → e_USD`.

> This is the entire mechanism. Everything else is making steps 2–3 fast (precomputed index, cached masks) and making the automaton powerful enough (CFG/pushdown for nesting).

---

<a name="8-the-full-trace"></a>
## 8. The Full Token-by-Token Trace

Target object (a *valid* one — it includes the required `item`):

```
{"supplier":"Acme","item":"bolts","quantity":200,"currency":"USD"}
```

Each row: the token the model emits, the characters it appends, the FSM state afterward, and the highlight of what the mask allowed at that step.

| # | Token (id) | Chars appended | Output so far | FSM state after | Mask highlight (legal set at this step) |
|---:|---|---|---|---|---|
| 1 | `{"` (6) | `{"` | `{"` | NEED_KEY `supplier` | only `{` or `{"` legal from START |
| 2 | `supplier` (9) | `supplier` | `{"supplier` | AFTER_KEY1 | only `supplier` (the one required first key) |
| 3 | `":"` (7) | `":"` | `{"supplier":"` | SUP_BODY | `":"`, `":`, `"` — the separators that open the value |
| 4 | `Acme` (14) | `Acme` | `...:"Acme` | SUP_BODY | any non-quote content **or** a closing quote |
| 5 | `","` (8) | `","` | `..."Acme","` | NEED_KEY `item` | close string, comma, open next key (boundary cross!) |
| 6 | `item` (10) | `item` | `...,"item` | AFTER_KEY2 | only `item` (required, ordered) |
| 7 | `":"` (7) | `":"` | `...item":"` | ITEM_BODY | value-opening separators |
| 8 | `bolts` (16) | `bolts` | `..."bolts` | ITEM_BODY | content or closing quote |
| 9 | `","` (8) | `","` | `..."bolts","` | NEED_KEY `quantity` | close, comma, open next key |
| 10 | `quantity` (11) | `quantity` | `...,"quantity` | AFTER_KEY3 | only `quantity` |
| 11 | `":` (29) | `":` | `..."quantity":` | **INT0** | `":` / `"` then `:` — note: no opening quote, value is a number |
| 12 | `200` (24) | `200` | `...:200` | **INT1** | `1,2,5,20,200` legal; **`0` illegal** as first digit |
| 13 | `,"` (30) | `,"` | `...200,"` | OPT_KEY | `}` **or** `,"` ; then key ∈ {`currency`,`delivery_date`} |
| 14 | `currency` (12) | `currency` | `...,"currency` | AFTER_KEYc | `currency` or `delivery_date` (we took currency) |
| 15 | `":"` (7) | `":"` | `...currency":"` | **ENUM0** | value separators → enter enum trie |
| 16 | `USD` (17) | `USD` | `..."USD` | e_USD | **only** `USD`,`EUR`,`INR`,`US` legal (the collapse) |
| 17 | `"` (3) | `"` | `..."USD"` | AFTER_CUR | only closing `"` |
| 18 | `}` (2) | `}` | `..."USD"}` | **ACCEPT** | `}` **or** `,"delivery_date":"` |
| 19 | `<EOS>` (0) | — | (done) | END | `<EOS>` legal *only because* state is ACCEPT |

Final string:

```
{"supplier":"Acme","item":"bolts","quantity":200,"currency":"USD"}
```

It parses. It was never *checked* after the fact — it was **impossible to build any other way**.

A few rows deserve a closer look:

- **Step 5 (`","`)** appended three characters that span a structural boundary: the closing quote of `supplier`'s value, the comma, and the opening quote of the next key. The DFA processed them as the character stream `"`,`,`,`"` and moved cleanly from "inside supplier string" to "expecting the `item` key." See Section 10.
- **Step 12 (`200`)**: from `INT0`, the token `0` was masked out (probability 0) but `200`, `20`, `2`, `1`, `5` were allowed. The model could not begin the quantity with a zero.
- **Step 16 (`USD`)**: the mask was the 4-token set from Section 6.2. Had the model preferred to write `"Acme"` again, it could not.
- **Step 19 (`<EOS>`)**: end-of-sequence is itself a masked token. It is legal **only** in an accept state. At step 17 (`AFTER_CUR`, not accepting) `<EOS>` was masked to −inf, so the model could not stop early and leave you with `{"supplier":"Acme",...,"currency":"USD` (no closing quote/brace).

---

<a name="9-enum-collapse"></a>
## 9. Zoom In: How the Enum Collapses

"The enum collapses to only legal continuations" is worth making fully precise, because it is the most visually obvious payoff.

At `ENUM0` the legal token set is `{USD, EUR, INR, US}`. Suppose the model samples `US` (token 20) instead of the whole `USD`. The FSM advances to `e_US`. Now recompute the mask for `e_US`:

```
allowed[e_US]:  walk every token from state e_US
   D   (21) :  D → e_USD        ✅   (completes "USD")
   USD (17) :  U → no edge from e_US   ❌
   "   (3)  :  " → no edge (we're mid-word, "US" alone is not a valid enum) ❌
   ...everything else...                ❌

allowed[e_US] = { 21 : e_USD }
```

From `e_US` the **only** legal token in the entire 31-token vocabulary is `D`. The model is now *forced* to complete `USD`. Then from `e_USD` the only legal token is the closing `"`. So the two tokenizations of the same value —

```
path A:  ENUM0 --USD--> e_USD --"--> AFTER_CUR     (one token for the value)
path B:  ENUM0 --US--> e_US --D--> e_USD --"--> AFTER_CUR   (two tokens)
```

— both land in exactly the same place. The grammar is enforced on **characters**, so it is completely indifferent to how the tokenizer happened to chop the word up. That indifference is the subject of the next section.

---

<a name="10-tokenizer-alignment"></a>
## 10. Zoom In: Tokenizer Alignment and Boundary-Crossing Tokens

This is the subtle part that trips people up. The grammar speaks **characters**; the model speaks **tokens**; and a single token can:

1. **span several grammar characters** (`supplier` = 8 chars, `200` = 3 chars), or
2. **cross a structural boundary** (`","` = end-of-value + separator + start-of-next-key), or
3. **be only a partial match** of a grammar element (`US` is half of `USD`).

All three are handled by the *same* mechanism from Section 6: when precomputing `allowed[s]`, we walk **every character of the token** through the DFA and keep the token only if the whole walk stays alive. The token's destination state is wherever the last character lands.

### 10.1 Boundary crossing, worked: token `","` from `SUP_BODY`

We are inside the `supplier` value (state `SUP_BODY`, which loops on non-quote characters and accepts a closing `"`). Walk token `","` (characters `"`, `,`, `"`):

```
SUP_BODY  --"-->  (supplier string closed; regex now expects  ,"item":" )
          --,-->  (comma consumed; expects the opening " of "item")
          --"-->  NEED_KEY_item   (quote consumed; expects the literal  item )
```

All three characters had live transitions, so `","` is admitted from `SUP_BODY` and lands in `NEED_KEY_item`. One token moved us across the value→separator→key boundary, and the FSM tracked it character by character without ever needing the tokenizer's boundaries to line up with the grammar's boundaries.

### 10.2 Why naive "token-level grammars" fail

If you tried to define the grammar over *tokens* instead of characters, you would have to enumerate every way the tokenizer might split every literal — `USD` vs `US`+`D`, `200` vs `2`+`0`+`0` vs `20`+`0`, `{"` vs `{`+`"`. That combinatorial mess is exactly what the character-DFA-plus-precomputed-index avoids. You define the language once over characters; the index automatically discovers which token splits are viable from each state.

### 10.3 The partial-token / "token healing" case

Token `US` (20) is not a complete enum value, yet it is allowed from `ENUM0` because it is a live **prefix**. The FSM simply moves to `e_US`, an intermediate (non-accepting) state, and the constraint that the value must finish as `USD` is carried forward by `allowed[e_US] = {D}`. There is no special "healing" logic — partial matches are just non-accepting states with their own (often tiny) allowed sets.

---

<a name="11-the-guarantee"></a>
## 11. Why It Is Guaranteed (the invariant)

The guarantee is not a heuristic; it is a short proof by induction on the generated tokens.

**Definitions.**
- A DFA state is **live** if at least one path from it reaches an accept state. (Dead states are pruned during compilation, so every state in the index is live.)
- The **invariant**: after emitting tokens `t1...tk`, the concatenated character string is a **prefix of some string in the schema's language**, and the FSM sits in the live state reached by that string.

**Base case.** Before any token, the string is empty and the FSM is at `START`, which is live (the schema admits at least one object). Invariant holds.

**Inductive step.** Assume the invariant after `t1...tk`, FSM in live state `s`. The mask permits only tokens in `allowed[s]`, i.e. tokens whose characters trace a path from `s` to some state `s'`. Two facts:
- `s'` is in the index, hence **live** (compilation removed dead states). So from `s'` an accept state is still reachable → the new string is still a valid prefix.
- Tokens **not** in `allowed[s]` get logit −∞ → probability 0 → cannot be sampled.

Therefore after `t(k+1)` the invariant still holds.

**Termination.** `<EOS>` is itself masked: it is in `allowed[s]` **iff** `s` is an accept state. So the model can only stop when the string so far is a *complete* member of the language — never in the middle of a value, never with an unclosed brace. The first moment generation can legally end, the string is guaranteed to be in the language, i.e. to satisfy the schema. ∎

> The probability of producing invalid JSON is therefore not "small" — it is **exactly zero**, because invalid tokens are assigned probability zero at every step.

---

<a name="12-real-libraries"></a>
## 12. The Real Libraries: Outlines, XGrammar, lm-format-enforcer

vLLM does not implement this itself; it plugs in a **guided-decoding backend** that produces the per-step mask. You select one with `guided_decoding_backend` (e.g. `"outlines"`, `"xgrammar"`, `"lm-format-enforcer"`), and pass the constraint via `GuidedDecodingParams(json=po_schema)` (also `choice=`, `regex=`, `grammar=`).

```
            ┌──────────────── vLLM engine ────────────────┐
 schema ───▶│  guided-decoding backend                    │
            │   • Outlines       (regex → FSM index)       │
            │   • XGrammar       (CFG → pushdown + cache)   │──▶ token mask (length |V|)
            │   • lm-format-enf. (token-by-token filter)    │        │
            └───────────────────────────────────────────────┘        ▼
                                                       logits + mask → sample
```

| Backend | Automaton | Key idea | Best for |
|---|---|---|---|
| **Outlines** (Willard & Louf 2023) | regex → DFA | Precompute the `state → {token → next state}` **index** so each step is an O(1) lookup. The paper that made this practical. | regex and flat/most JSON schemas |
| **XGrammar** | context-free grammar → pushdown automaton | Splits tokens into **context-independent** (decidable from the grammar alone, cached once) vs **context-dependent** (checked against the stack at run time); precomputed **adaptive token mask cache**; overlaps mask building with the GPU forward pass so overhead ≈ 0. Now the vLLM default for many cases. | nested/recursive JSON, full CFGs, speed |
| **lm-format-enforcer** | token-prefix filter | Walks the allowed token tree, permits optional-whitespace flexibility and "beam"-style choices; less rigid than a pure DFA. | when you want some formatting slack |

All three answer the same question — "which tokens are legal next?" — and feed vLLM the same artifact: a length-`|V|` mask added to the logits before softmax. The differences are in **expressive power** (regex vs CFG) and **how aggressively they precompute / cache** to keep the per-step cost negligible.

The shared insight is the **precomputed transition map**: the expensive automaton work is done once at compile time, so the hot path that runs on every one of potentially thousands of tokens is just *look up the set, add the mask, sample*.

---

<a name="13-scale"></a>
## 13. Scale: Toy vs Production

The mask is a vector with one entry per vocabulary token. Its width is the only thing that grows with a real tokenizer; the algorithm is unchanged.

| Quantity | Toy | GPT-2 | Llama 3 |
|---|---:|---:|---:|
| Vocabulary size `|V|` | 31 | 50,257 | 128,256 |
| Mask size (1 byte/token) | 31 B | 49 KB | 125 KB |
| Mask as a bitmask (1 bit/token) | 4 B | 6.3 KB | 16 KB |
| FSM states for the PO schema (order of magnitude) | ~30 | ~hundreds | ~hundreds |
| Index entries `|states| x |V|` (dense, upper bound) | ~930 | ~25 M | ~64 M |

Two engineering consequences:

- **Storage.** A dense index of `states x vocab` is large; real backends store it **sparsely** (most tokens are illegal in most states) and/or compute context-dependent rows on the fly with a cache.
- **Latency.** The mask must be ready before each sampling step. XGrammar's headline trick is overlapping mask construction with the model's forward pass on the GPU, driving the added latency toward zero so constrained decoding is essentially free per token.

```
Per-step work (independent of |V| growth except mask width):
   look up allowed[state]      O(1) amortized (precomputed/cached)
   write mask                  O(|V|)   (a memset + scatter; microseconds)
   add mask to logits          O(|V|)   (already part of the sampler)
   advance state               O(1)
```

> Compare to the alternative — generate freely, `json.loads()`, and **retry on failure**. Retrying re-runs the whole forward pass (the expensive part) an unbounded number of times and still offers no guarantee. Masking pays a fixed microsecond cost per token and guarantees success on the first try.

---

<a name="14-limitations"></a>
## 14. Limitations and Gotchas

- **Valid ≠ correct.** The FSM guarantees the output *parses and matches the schema*. It does **not** guarantee the values are *right*: it will happily emit `"quantity":1` to stay legal if the legal-but-low-effort path is what gets sampled. Constrain the **shape**; verify the **values** with evals and business-rule guardrails.
- **Distribution distortion.** Masking renormalizes over legal tokens, which can nudge the model off the path it "wanted." If the model strongly wanted an illegal token, forcing a legal one can degrade quality. Good schemas leave room (don't over-constrain) and good prompts make the legal path also the natural one.
- **Property ordering.** Our regex fixed the key order (`supplier, item, quantity, ...`). Real emitters either fix the order (simplest) or build a larger automaton allowing permutations (`additionalProperties:false` still bounds it). Permutations blow up state count, which is one reason CFG backends (XGrammar) scale better than pure regex here.
- **Unbounded / recursive schemas** (deeply nested objects, arrays of arrays) exceed regular languages — you need the CFG/pushdown backend, not a plain DFA.
- **Compile cost.** Building the index for a brand-new, large schema has a one-time latency (often cached by schema hash). The first request with a novel schema can be slower than subsequent ones.
- **Tokenizer quirks.** Byte-level tokenizers, special tokens, and leading-space tokens (`▁`) must be reconciled with the character DFA; backends handle this, but a mismatched tokenizer config is a common source of "why is nothing allowed here?" bugs.

---

<a name="15-summary"></a>
## 15. Summary

```
JSON Schema
   │  Stage 1: compile to a regex (or CFG for nesting)
   ▼
Regex / Grammar
   │  Stage 2: build a character-level DFA (or pushdown automaton)
   ▼
DFA  (states + character transitions; dead states pruned ⇒ every state is "live")
   │  Stage 3: precompute the INDEX  state → { token_id → next_state }
   ▼
Index  ───────────────────────── used every decode step ─────────────────────────┐
                                                                                   ▼
Run time, per token:  logits ─► mask illegal (−∞) using index[state] ─► softmax ─► sample ─► advance state
```

The five things to remember:

1. **The question** every step asks is "which tokens are legal now?", and the answer comes from a finite-state machine compiled from your schema.
2. **The mask** sets illegal tokens' logits to −∞, so after softmax they have probability **exactly 0**. The model cannot emit them.
3. **The index** (Outlines' contribution) precomputes `state → {token → next state}` so the per-step cost is a lookup, not a re-parse.
4. **Characters, not tokens**, are the unit of the grammar — which is why multi-character and boundary-crossing tokens (`{"`, `","`, `200`, `US`+`D`) all just work, and why the enum can only ever complete as `USD`/`EUR`/`INR`.
5. **The guarantee** is a proven invariant: every emitted token keeps the FSM in a live state, and `<EOS>` is legal only at an accept state — so the finished string is always in the schema's language.

That is the whole backend: a tiny state machine standing between the model's logits and the sampler, quietly deleting every token that would have broken your JSON.
