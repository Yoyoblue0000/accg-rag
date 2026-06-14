"""全面检查图中 CLASS 相关的所有边（CONTAINS / INHERITS / CALLS）"""
import sys
from collections import defaultdict, deque

from accg.builder import GraphBuilder
from accg.models import EdgeType, NodeType


def check(graph):
    nodes = dict(graph.nodes(data=True))
    issues = []
    class_ids = {nid for nid, d in nodes.items()
                 if d.get("node_type") in (NodeType.CLASS, NodeType.EXTERNAL)}

    inherits   = defaultdict(list)  # child -> [parent]
    contains   = defaultdict(list)  # class -> [method/nested]
    calls_to_cls  = defaultdict(list)  # target_class -> [(caller, edge)]
    calls_to_init = defaultdict(list)  # target_init -> [(caller, edge)]
    init_to_class = {}                # __init__ node_id -> class node_id

    for u, v, d in graph.edges(data=True):
        et = d.get("edge_type")
        if et == EdgeType.INHERITS:
            inherits[u].append(v)
        elif et == EdgeType.CONTAINS:
            if u in class_ids:
                contains[u].append(v)
                if nodes[v].get("name") == "__init__":
                    init_to_class[v] = u
        elif et == EdgeType.CALLS:
            if v in class_ids:
                calls_to_cls[v].append((u, d))
            if nodes[v].get("name") == "__init__":
                calls_to_init[v].append((u, d))
                if v not in init_to_class:
                    init_to_class[v] = None

    # === 1. INHERITS ===
    for child, parents in inherits.items():
        if child not in class_ids:
            issues.append(f"[INHERITS] src not CLASS: {child}")
        for p in parents:
            if p not in class_ids:
                issues.append(f"[INHERITS] tgt not CLASS: {child} -> {p}")
            if p == child:
                issues.append(f"[INHERITS] self-loop: {child}")
            if p not in nodes:
                issues.append(f"[INHERITS] orphan: {child} -> {p}")

    for start in class_ids:
        visited = set()
        queue = deque([(start, [start])])
        while queue:
            cur, path = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            for parent in inherits.get(cur, []):
                if parent == start:
                    issues.append(f"[INHERITS] cycle: {' -> '.join(path)} -> {parent}")
                elif parent not in visited:
                    queue.append((parent, path + [parent]))

    # === 2. CONTAINS ===
    for cls_id, method_list in contains.items():
        for m_id in method_list:
            if m_id not in nodes:
                issues.append(f"[CONTAINS] orphan: {cls_id} -> {m_id}")
            else:
                nt = nodes[m_id].get("node_type")
                if nt not in (NodeType.METHOD, NodeType.CLASS):
                    issues.append(f"[CONTAINS] bad type {nt}: {cls_id} -> {m_id}")

    # === 3. CALLS -> CLASS (should be class_construct_class only) ===
    for cls_id, callers in calls_to_cls.items():
        for caller_id, edge in callers:
            if caller_id not in nodes:
                issues.append(f"[CALLS->CLASS] orphan caller: {caller_id}")
            strat = edge.get("strategy", "")
            if strat != "class_construct_class":
                issues.append(f"[CALLS->CLASS] bad strategy {strat}: {caller_id} -> {cls_id}")

    # === 4. CALLS -> __init__ (allow class_construct / import_map / super_resolve / ...) ===
    CLASS_STRATEGIES = {"class_construct", "class_construct_class"}
    for init_id, callers in calls_to_init.items():
        cls_id = init_to_class.get(init_id)
        for caller_id, edge in callers:
            if caller_id not in nodes:
                issues.append(f"[CALLS->__init__] orphan caller: {caller_id}")
            strat = edge.get("strategy", "")
            # class_construct_class on __init__ is wrong (should be on CLASS node)
            if strat == "class_construct_class":
                issues.append(f"[CALLS->__init__] bad strategy class_construct_class: {caller_id} -> {init_id}")
        if cls_id is None:
            issues.append(f"[CALLS->__init__] __init__ not in any CLASS CONTAINS: {init_id}")
        elif init_id not in contains.get(cls_id, []):
            issues.append(f"[CALLS->__init__] __init__ not in CONTAINS of {cls_id}: {init_id}")

    # === 5. Double-edge: each class-construct caller must have both CLASS + __init__ edges ===
    caller_class_edges = defaultdict(set)
    caller_init_edges  = defaultdict(set)
    for cls_id, callers in calls_to_cls.items():
        for caller_id, _edge in callers:
            if _edge.get("strategy") in CLASS_STRATEGIES:
                caller_class_edges[caller_id].add(cls_id)
    for init_id, callers in calls_to_init.items():
        for caller_id, _edge in callers:
            if _edge.get("strategy") in CLASS_STRATEGIES:
                caller_init_edges[caller_id].add(init_id)

    # Check class->init mapping for own __init__ (not just inherited)
    for cls_id in class_ids:
        own_init = f"{cls_id}::__init__"
        if own_init in nodes:
            init_to_class.setdefault(own_init, cls_id)

    for caller_id, init_set in caller_init_edges.items():
        class_set = caller_class_edges.get(caller_id, set())
        for init_id in init_set:
            owner_cls = init_to_class.get(init_id)
            if owner_cls is None:
                continue
            # Find all classes in the chain that this init belongs to
            related_classes = {owner_cls}
            for cls_id, parents in inherits.items():
                visited = set()
                q = deque(parents)
                while q:
                    p = q.popleft()
                    if p in visited:
                        continue
                    visited.add(p)
                    if p == owner_cls:
                        related_classes.add(cls_id)
                        break
                    q.extend(inherits.get(p, []))
            if not (class_set & related_classes):
                issues.append(f"[double-edge] {caller_id} has init edge to {init_id} but no CLASS edge to any of {related_classes}")

    # === 6. Stats ===
    cls_with_own_init = sum(1 for c in class_ids if f"{c}::__init__" in nodes)
    cls_without_init = len(class_ids) - cls_with_own_init
    print(f"CLASS nodes: {len(class_ids)} (own __init__: {cls_with_own_init}, no __init__: {cls_without_init})")
    print(f"INHERITS: {sum(len(v) for v in inherits.values())}")
    print(f"CONTAINS: {sum(len(v) for v in contains.values())}")
    print(f"CALLS->CLASS: {sum(len(v) for v in calls_to_cls.values())}")
    print(f"CALLS->__init__: {sum(len(v) for v in calls_to_init.values())}")
    print()

    if issues:
        print(f"{'='*60}")
        print(f"Issues: {len(issues)}")
        print(f"{'='*60}")
        for i in issues:
            print(f"  {i}")
    else:
        print("All CLASS edge checks passed.")

    return len(issues)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test_repos/requests_repo"
    builder = GraphBuilder()
    graph = builder.build(path)
    exit(check(graph))
