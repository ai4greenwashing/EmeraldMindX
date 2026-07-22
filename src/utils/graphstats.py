#!/usr/bin/env python3
"""
Memory-efficient shortest path calculation using Neo4j GDS
Avoids memory issues by using GDS algorithms instead of Cypher pattern matching
"""

from neo4j import GraphDatabase
import statistics

# Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "emeraldmind"
GRAPH_NAME = "emeraldGraph"

def calculate_shortest_paths_gds(sample_size=500):
    """
    Calculate shortest paths using GDS algorithms (memory-efficient)
    Uses BFS from sampled source nodes instead of all-pairs shortest path
    """
    print("="*60)
    print("MEMORY-EFFICIENT SHORTEST PATH CALCULATION")
    print("="*60)
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Drop existing projection
            try:
                session.run(f"CALL gds.graph.drop('{GRAPH_NAME}', false)")
                print("Dropped existing projection")
            except:
                pass
            
            # Create projection
            print("\nCreating graph projection...")
            result = session.run("""
                CALL gds.graph.project($graphName, '*', '*')
                YIELD graphName, nodeCount, relationshipCount
                RETURN nodeCount, relationshipCount
            """, graphName=GRAPH_NAME).single()
            
            node_count = result['nodeCount']
            rel_count = result['relationshipCount']
            print(f"  Nodes: {node_count:,}")
            print(f"  Relationships: {rel_count:,}")
            
            # Find largest component
            print("\nFinding largest component...")
            result = session.run(f"""
                CALL gds.wcc.stream('{GRAPH_NAME}')
                YIELD nodeId, componentId
                WITH componentId, count(*) AS size
                ORDER BY size DESC
                LIMIT 1
                RETURN componentId AS largestComp, size AS componentSize
            """).single()
            
            largest_comp = result['largestComp']
            comp_size = result['componentSize']
            print(f"  Largest component: {comp_size:,} nodes (ID: {largest_comp})")
            
            if comp_size < 2:
                print("\n✗ Component too small for path calculation")
                return None
            
            # Get sample of nodes from largest component
            print(f"\nSampling {sample_size} nodes from largest component...")
            result = session.run(f"""
                CALL gds.wcc.stream('{GRAPH_NAME}')
                YIELD nodeId, componentId
                WHERE componentId = {largest_comp}
                WITH nodeId
                LIMIT {sample_size}
                RETURN collect(nodeId) AS sampleNodes
            """).single()
            
            sample_nodes = result['sampleNodes']
            actual_sample = len(sample_nodes)
            print(f"  Sampled {actual_sample} nodes")
            
            # Use BFS from each sampled node to calculate paths
            # This is MUCH more memory efficient than all-pairs
            print(f"\nCalculating BFS distances from {actual_sample} source nodes...")
            print("  (This streams results, avoiding memory issues)")
            
            all_distances = []
            max_distance = 0
            min_distance = float('inf')
            paths_found = 0
            
            # Process in batches to show progress
            batch_size = 50
            for i in range(0, actual_sample, batch_size):
                batch_nodes = sample_nodes[i:i+batch_size]
                
                # Run BFS from each node in batch
                result = session.run(f"""
                    UNWIND $sourceNodes AS sourceId
                    CALL gds.bfs.stream('{GRAPH_NAME}', {{
                        sourceNode: sourceId,
                        maxDepth: 20
                    }})
                    YIELD path
                    WITH size(nodes(path)) - 1 AS distance
                    WHERE distance > 0
                    RETURN distance
                """, sourceNodes=batch_nodes)
                
                # Stream results without loading all into memory
                batch_count = 0
                for record in result:
                    dist = record['distance']
                    all_distances.append(dist)
                    max_distance = max(max_distance, dist)
                    min_distance = min(min_distance, dist)
                    batch_count += 1
                
                paths_found += batch_count
                print(f"  Processed {min(i+batch_size, actual_sample)}/{actual_sample} sources... ({paths_found:,} paths so far)")
            
            # Calculate statistics
            if paths_found > 0:
                avg_distance = statistics.mean(all_distances)
                median_distance = statistics.median(all_distances)
                
                print("\n" + "="*60)
                print("SUCCESS - Shortest Path Statistics:")
                print("="*60)
                print(f"  Average Shortest Path: {avg_distance:.2f}")
                print(f"  Median Shortest Path: {median_distance:.2f}")
                print(f"  Diameter (max distance): {max_distance}")
                print(f"  Minimum Path: {min_distance}")
                print(f"  Total Paths Calculated: {paths_found:,}")
                print(f"  Sample Size: {actual_sample} source nodes")
                print("="*60)
                
                sp_stats = {
                    'avgShortestPath': avg_distance,
                    'medianShortestPath': median_distance,
                    'diameter': max_distance,
                    'minPath': min_distance,
                    'pathsCalculated': paths_found,
                    'sampleSize': actual_sample,
                    'method': 'GDS BFS (memory-efficient)'
                }
            else:
                print("\n✗ No paths found")
                sp_stats = None
            
            # Cleanup
            print("\nCleaning up...")
            session.run(f"CALL gds.graph.drop('{GRAPH_NAME}', false)")
            print("Dropped projection")
            
            return sp_stats
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to cleanup
        try:
            driver.session().run(f"CALL gds.graph.drop('{GRAPH_NAME}', false)")
        except:
            pass
        
        return None
    finally:
        driver.close()


def main():
    # Calculate with default sample size
    stats = calculate_shortest_paths_gds(sample_size=60000)
    
    if stats:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Average Shortest Path Length: {stats['avgShortestPath']:.2f}")
        print(f"Graph Diameter: {stats['diameter']}")
        print("="*60)
        
        # Save to file
        import json
        from pathlib import Path
        from datetime import datetime
        
        output_dir = Path("graph_stats3")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"shortest_paths_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nSaved to: {output_file}")

if __name__ == "__main__":
    main()