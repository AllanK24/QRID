import onnx, re
from collections import deque

def yolo_head_nodes_to_skip(model_path: str):
    """
    Return names of YOLO post-processing nodes (Sigmoid / Softmax / DFL maths)
    that should stay FP32 during static INT8 quantisation.
    """
    m = onnx.load(model_path)

    # --- 1️⃣ build helper maps ---
    prod = {o: n for n in m.graph.node for o in n.output}          # tensor → node
    outs = [prod[o.name] for o in m.graph.output if o.name in prod] # producer nodes

    # --- 2️⃣ infer the *head scope* prefix (e.g. '/model.22') ---
    def head_scope(name: str) -> str:
        m = re.search(r"/model\.\d+", name)          # Ultralytics naming
        if m:
            return m.group(0)                        # '/model.22'
        # fall back: take the first two path components
        parts = name.split('/')
        return '/'.join(parts[:2])

    prefixes = {head_scope(n.name) for n in outs}

    # --- 3️⃣ BFS inside the head, stop at the first non-DFL Conv ---
    queue, visited, skip = deque(outs), set(), set()

    while queue:
        node = queue.popleft()
        if node.name in visited:
            continue
        visited.add(node.name)

        # work only on nodes that share the head prefix
        if not any(node.name.startswith(p) for p in prefixes):
            continue

        # decide whether to _skip_ this node
        if (
            node.op_type in ("Sigmoid", "Softmax")               # logits → probs
            or "/dfl/" in node.name                              # DFL branch
            or node.op_type in ("Add", "Sub", "Mul", "Div",
                               "Concat", "Split", "Slice",
                               "Reshape", "Transpose")
        ):
            skip.add(node.name)

        # if we reach a Conv that is _not_ in the DFL path, stop tracing back
        if node.op_type == "Conv" and "/dfl/" not in node.name:
            continue

        # otherwise enqueue the producers of this node’s inputs
        for inp in node.input:
            pred = prod.get(inp)
            if pred and pred.name not in visited:
                queue.append(pred)

    return sorted(skip)