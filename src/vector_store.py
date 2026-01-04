"""Vector store wrapper placeholder."""

class VectorStore:
    def __init__(self):
        self.store = []

    def add(self, vector, meta=None):
        self.store.append((vector, meta))

    def query(self, vector, top_k=5):
        return self.store[:top_k]
