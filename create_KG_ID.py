import os
import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase


class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        query = "MATCH (n) DETACH DELETE n"
        with self.driver.session() as session:
            session.run(query)

    def create_node(self, name, node_type):
        query = f"""
            MERGE (n:{node_type} {{name: $name}})
            RETURN elementId(n) as node_id
        """
        with self.driver.session() as session:
            result = session.run(query, name=name)
            return result.single()["node_id"]

    def create_relationship(self, start_id, end_id, relationship):
        query = f"""
            MATCH (a), (b)
            WHERE elementId(a) = $start_id AND elementId(b) = $end_id
            CREATE (a)-[r:{relationship}]->(b)
        """
        with self.driver.session() as session:
            session.run(query, start_id=start_id, end_id=end_id)

# Function to generate a unique file name
def get_unique_filename(base_name):
    count = 1
    file_name = base_name
    while os.path.exists(file_name):
        file_name = f"{base_name.rsplit('.', 1)[0]}_{count:02d}.csv"
        count += 1
    return file_name

# Separate and extract id
def parse_element_id(element_id):
    # 示例：从 "4:5f8a6b-1" 中提取数字部分
    return int(element_id.split(":")[-1])

# Main program
def main():
    # Connect to your Neo4j database
    uri = "bolt://localhost:7687"  # Adjust if running on a remote server
    user = "username"
    password = "*******"
    kg = KnowledgeGraph(uri, user, password)

    # Load data from Excel file
    file_path = "kg_data.xlsx"  # Update this with your actual file path
    sheet_data = pd.read_excel(file_path)

    # Clear the database before importing new data
    kg.clear_database()

    # Keep track of created nodes to avoid duplicates
    created_nodes = {}

    # List for exporting CSV data
    export_data = []

    # Define type mappings
    type_mapping = {"机器": 0, "构型": 1, "动型": 2, "控型": 3, "优型": 4}
    relationship_mapping = {
        row["关系"]: row["关系序号"] for _, row in sheet_data.dropna(subset=["关系", "关系序号"]).iterrows()
    }

    start_id_lst = []
    # Iterate over the DataFrame to create nodes and relationships
    for _, row in tqdm(sheet_data.iterrows(), total=sheet_data.shape[0], desc="Processing rows"):
        start_name = row['首节点名称']
        start_type = row['首节点类型']
        end_name = row['尾节点名称']
        end_type = row['尾节点类型']
        relationship = row['关系']
        machine_name = row['机器'] if '机器' in row else None
        machine_relationship = row['机器关系'] if '机器关系' in row else None

        # Create or get ID for start node
        if pd.notna(start_name) and pd.notna(start_type):
            if (start_name, start_type) not in created_nodes:
                start_id = kg.create_node(start_name, start_type)
                created_nodes[(start_name, start_type)] = start_id
            else:
                start_id = created_nodes[start_name, start_type]


        # Create or get ID for end node
        if pd.notna(end_name) and pd.notna(end_type):
            if (end_name, end_type) not in created_nodes:
                end_id = kg.create_node(end_name, end_type)
                created_nodes[(end_name, end_type)] = end_id
            else:
                end_id = created_nodes[(end_name, end_type)]

        # Create relationship and export data
        if pd.notna(start_name) and pd.notna(end_name) and pd.notna(relationship):
            relationship_id = relationship_mapping.get(relationship, 0)
            kg.create_relationship(start_id, end_id, relationship)
            start_num = parse_element_id(start_id)
            end_num = parse_element_id(end_id)
            export_data.append([
                start_num, end_num, int(relationship_id),
                type_mapping.get(start_type, 0), type_mapping.get(end_type, 0)
            ])

        # Handle machine-related nodes and relationships
        if pd.notna(machine_name):
            machine_type = "机器"
            if (machine_name, machine_type) not in created_nodes:
                machine_id = kg.create_node(machine_name, machine_type)
                created_nodes[(machine_name, machine_type)] = machine_id
            else:
                machine_id = created_nodes[(machine_name, machine_type)]

            if pd.notna(start_name) and pd.notna(machine_relationship):
                machine_relationship_id = relationship_mapping.get(machine_relationship, 0)
                if start_id not in start_id_lst:
                    kg.create_relationship(machine_id, start_id, machine_relationship)
                    start_id_lst.append(start_id)
                    start_num = parse_element_id(start_id)
                    machine_num = parse_element_id(machine_id)
                    export_data.append([
                        machine_num, start_num, int(machine_relationship_id),
                        type_mapping.get(machine_type, 0), type_mapping.get(start_type, 0)
                    ])

    # Export data to CSV
    export_df = pd.DataFrame(export_data, columns=None)
    export_file_name = get_unique_filename("exported_data.dat")
    export_df.to_csv(export_file_name, index=False, header=False, encoding="utf-8-sig")

    # Close the connection
    kg.close()

if __name__ == "__main__":
    main()
