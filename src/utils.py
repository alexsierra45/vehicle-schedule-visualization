import json
from pyomo.environ import value

def save_solution_json(model, path="solution.json"):
    sol = {}

    sol["N"] = [
        {"t": str(t), "c": str(c), "value": float(value(model.N[t,c]))}
        for t in model.T for c in model.C
    ]

    sol["w"] = [
        {"k": int(k), "m": str(m), "value": float(value(model.w[k,m]))}
        for k in model.K for m in model.M
    ]

    sol["SoC"] = [
        {"k": int(k), "h": int(h), "value": float(value(model.SoC[k,h]))}
        for k in model.K for h in model.H
    ]

    sol["z"] = [
        {"k": int(k), "t": str(t), "c": str(c), "h": int(h), "value": float(value(model.z[k,t,c,h]))}
        for k in model.K for t in model.T for c in model.C for h in model.H
    ]

    if hasattr(model, "p"):
        sol["p"] = [
            {"k": int(k), "t": str(t), "c": str(c), "h": int(h), "value": float(value(model.p[k,t,c,h]))}
            for k in model.K for t in model.T for c in model.C for h in model.H
        ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sol, f, ensure_ascii=False)

    print(f"Solution saved to {path}")
