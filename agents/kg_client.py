"""
星学伴 - 知识图谱查询模块
用法：from kg_client import KGClient; kg = KGClient(); kg.get_prerequisites("一元二次方程")
"""
from neo4j import GraphDatabase
from typing import List, Dict, Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class KGClient:
    """Neo4j AuraDB 知识图谱客户端"""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        # 从 URI 提取数据库名
        self.db = NEO4J_URI.split("://")[1].split(".")[0] if "://" in NEO4J_URI else "neo4j"

    def close(self):
        self.driver.close()

    def _run(self, query, **params):
        with self.driver.session(database=self.db) as s:
            return list(s.run(query, **params))

    # ========== Agent 会用到的 6 个查询 ==========

    def find_kp(self, name: str) -> Optional[Dict]:
        """① 通过名称模糊查找知识点 → 返回结构化 ID"""
        r = self._run("""
            MATCH (n:Node {type: 'knowledge_point'})
            WHERE n.name CONTAINS $name
            RETURN n.id AS id, n.name AS name, n.domain AS domain,
                   n.chapter AS chapter, n.chapter_name AS chapter_name
            LIMIT 5
        """, name=name)
        return r[0].data() if r else None

    def get_kp_by_id(self, kp_id: str) -> Optional[Dict]:
        """通过 ID 获取知识点详情"""
        r = self._run("""
            MATCH (n:Node {id: $id})
            RETURN n.name AS name, n.type AS type, n.domain AS domain,
                   n.chapter_name AS chapter_name, n.description AS description
        """, id=kp_id)
        return r[0].data() if r else None

    def get_prerequisites(self, kp_name: str) -> List[str]:
        """② 诊断根因：按学习顺序列出所有前驱知识点（从基础到当前）"""
        r = self._run("""
            MATCH path = (start:Node {name: $name})<-[:PREREQUISITE*]-(pre:Node {type: 'chapter'})
            WITH path ORDER BY length(path) DESC LIMIT 1
            RETURN [n in nodes(path) | n.name] AS chain
        """, name=kp_name)
        if r:
            chain = r[0]['chain']
            chain.reverse()  # 翻转为从基础到高级
            return chain
        return []

    def get_next_steps(self, kp_name: str) -> List[str]:
        """③ 组卷决策：列出后继章节（进阶方向）"""
        r = self._run("""
            MATCH (n:Node {name: $name})-[:PREREQUISITE]->(next:Node {type: 'chapter'})
            RETURN next.name AS name
        """, name=kp_name)
        return [x['name'] for x in r]

    def get_chapter_kps(self, chapter_num: int) -> List[Dict]:
        """④ 知识追踪：获取某章所有知识点清单"""
        r = self._run("""
            MATCH (n:Node {type: 'knowledge_point', chapter: $ch})
            RETURN n.id AS id, n.name AS name
            ORDER BY n.id
        """, ch=chapter_num)
        return [x.data() for x in r]

    def get_kps_by_domain(self, domain: str) -> List[Dict]:
        """⑤ 按领域查知识点"""
        r = self._run("""
            MATCH (n:Node {type: 'knowledge_point', domain: $domain})
            RETURN n.id AS id, n.name AS name, n.chapter AS chapter
        """, domain=domain)
        return [x.data() for x in r]

    def search(self, keyword: str, limit: int = 10) -> List[Dict]:
        """⑥ 搜索：模糊匹配知识点名称"""
        r = self._run("""
            MATCH (n:Node) WHERE n.type IN ['knowledge_point', 'section', 'chapter']
            AND (n.name CONTAINS $kw OR n.description CONTAINS $kw)
            RETURN n.id AS id, n.name AS name, n.type AS type, n.domain AS domain
            LIMIT $limit
        """, kw=keyword, limit=limit)
        return [x.data() for x in r]


# ========== 使用示例 ==========
if __name__ == "__main__":
    kg = KGClient()

    # ① 学生拍照识别出"二次函数顶点坐标"，查 KG 定位
    kp = kg.find_kp("二次函数")
    print("① 定位:", kp)

    # ② 学生老做错，Agent 诊断——从哪开始补？
    chain = kg.get_prerequisites("二次函数的图象及性质")
    print("② 前置链:", " → ".join(chain))

    # ③ 学生掌握了，下一步学什么？
    next_steps = kg.get_next_steps("一次方程（组）及其应用")
    print("③ 后继:", next_steps)

    # ④ 第7章有哪些知识点？（初始化知识追踪的 θ 值列表）
    kps = kg.get_chapter_kps(7)
    print(f"④ 第7章 KPs: {len(kps)} 个, 前3: {[k['name'] for k in kps[:3]]}")

    # ⑤ 数与代数领域有哪些？
    domain_kps = kg.get_kps_by_domain("数与代数")
    print(f"⑤ 数与代数: {len(domain_kps)} KPs")

    # ⑥ 模糊搜索
    results = kg.search("方程", limit=5)
    print(f"⑥ 搜索'方程': {[r['name'] for r in results]}")

    kg.close()
