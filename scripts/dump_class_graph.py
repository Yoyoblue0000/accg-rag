"""输出所有 CLASS 节点及其边，用于人工检查"""
from collections import defaultdict

from accg.builder import GraphBuilder
from accg.models import EdgeType, NodeType

g = GraphBuilder().build("test_repos/requests_repo")

class_by_file = defaultdict(list)
for nid, d in g.nodes(data=True):
    if d.get("node_type") == NodeType.CLASS:
        class_by_file[d.get("file_path", "?")].append((nid, d))

calls_cls = defaultdict(list)
calls_init = defaultdict(list)
for u, v, d in g.edges(data=True):
    if d.get("edge_type") != EdgeType.CALLS:
        continue
    vt = g.nodes[v].get("node_type")
    vn = g.nodes[v].get("name", "")
    if vt == NodeType.CLASS:
        calls_cls[v].append((u, d.get("strategy"), d.get("confidence")))
    if vn == "__init__":
        calls_init[v].append((u, d.get("strategy"), d.get("confidence")))

total_inh = sum(1 for _, _, d in g.edges(data=True) if d.get("edge_type") == EdgeType.INHERITS)
total_con = sum(1 for u, _, d in g.edges(data=True)
                if d.get("edge_type") == EdgeType.CONTAINS and g.nodes[u].get("node_type") == NodeType.CLASS)
total_cls_call = sum(len(v) for v in calls_cls.values())
total_init_call = sum(len(v) for v in calls_init.values())

print(f"=== CLASS 节点: {sum(len(v) for v in class_by_file.values())} | INHERITS: {total_inh} | CONTAINS: {total_con} | CALLS->CLASS: {total_cls_call} | CALLS->__init__: {total_init_call} ===\n")

for fname in sorted(class_by_file):
    classes = class_by_file[fname]
    print(f"## {fname} ({len(classes)} classes)")
    for nid, d in sorted(classes, key=lambda x: x[1].get("start_line", 0)):
        name = d.get("name")
        sl, el = d.get("start_line", 0), d.get("end_line", 0)
        bases = d.get("extra", {}).get("bases", [])
        has_init = f"{nid}::__init__" in g.nodes

        out_edges = []
        for _, v, ed in g.out_edges(nid, data=True):
            et = ed.get("edge_type")
            if et == EdgeType.CONTAINS:
                out_edges.append(f"CONTAINS->{g.nodes[v].get('name', v)}")
            elif et == EdgeType.INHERITS:
                out_edges.append(f"INHERITS->{g.nodes[v].get('name', v)}")

        in_cls = calls_cls.get(nid, [])
        in_init = calls_init.get(f"{nid}::__init__", [])

        init_mark = "+init" if has_init else "-init"
        bases_str = f"({', '.join(bases)})" if bases else ""
        print(f"  {name} {bases_str} [{sl}-{el}] {init_mark}")

        for e in sorted(out_edges):
            print(f"    OUT {e}")

        for caller, strat, conf in in_cls[:5]:
            cn = g.nodes[caller].get("name", caller)
            print(f"    IN CLASS  <- {cn} ({strat}, {conf})")
        if len(in_cls) > 5:
            print(f"    ... +{len(in_cls)-5} more CLASS callers")

        for caller, strat, conf in in_init[:5]:
            cn = g.nodes[caller].get("name", caller)
            print(f"    IN __init__ <- {cn} ({strat}, {conf})")
        if len(in_init) > 5:
            print(f"    ... +{len(in_init)-5} more __init__ callers")
        print()
