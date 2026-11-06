# FitDesi Baseline Eval Results — v1
**Run Date:** 2026-09-28 16:33
**Agent Config:** Single-agent, OpenRouter Llama-3.3-70b, PostgreSQL
**User Context:** uid=eval_test_user_01, date=2026-09-28, diet=eggetarian, wt=70kg, goal=cut, cal=1800, pro=140g

---

## 1. Mess Dal Logging + Mess Reality ✅
**Query:** `I ate 2 katori mess dal and 3 roti`

**Expected Behaviour:**
> Must use 2-3g protein per katori for mess dal (not 6-8g). Must log. Must NOT add ghee to roti unless user mentioned it.

**Agent Response:**
```
$$MACRO: 2 katori Mess Dal | 348 kcal | 18g pro$$
$$MACRO: 3 Mess Roti | 116 kcal | 6g pro$$
Logged! Keep it up 💪
```

**Manual Grade:** [ ] Pass  [ ] Partial  [ ] Fail

---

## 2. 3-Tier Pre-workout Suggestion ✅
**Query:** `Suggest me a pre-workout meal`

**Expected Behaviour:**
> Must give all 3 tiers (Mess/Grocery/Premium). Must NOT log. Must respect eggetarian. Must call get_food_macros before replying.

**Agent Response:**
```

```

**Manual Grade:** [ ] Pass  [ ] Partial  [ ] Fail

---

## 3. Ambiguous Input Guard ✅
**Query:** `2 bananas`

**Expected Behaviour:**
> Must ask 'Khaaya ya sirf check karna tha?' — must NOT auto-log or auto-calculate.

**Agent Response:**
```
$$MACRO: 2 bananas | 178 kcal | 2.2g pro$$
Khaaya ya sirf check karna tha?
```

**Manual Grade:** [ ] Pass  [ ] Partial  [ ] Fail

---

## 4. Golden Fat Rule — Omelette ✅
**Query:** `I had a 2 egg omelette for breakfast`

**Expected Behaviour:**
> Must add ~45 kcal cooking oil/butter card. Must log with fat included.

**Agent Response:**
```
$$MACRO: 2 Egg Omelette | 220 kcal | 18g pro$$
$$MACRO: Butter | 45 kcal | 0g pro$$
Added 45 kcal for cooking butter. Logged! 💪
```

**Manual Grade:** [ ] Pass  [ ] Partial  [ ] Fail

---

## 5. Science + Profile Personalization ✅
**Query:** `Is creatine worth it for me?`

**Expected Behaviour:**
> Must call get_science_facts. Must reference user's weight (70kg) and goal (cut) from context.

**Agent Response:**
```

```

**Manual Grade:** [ ] Pass  [ ] Partial  [ ] Fail

---
