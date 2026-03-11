"""
Module: scoring.py
Responsibility: Execute Kingdomino rules on the 5x5 data matrix.
"""

from typing import List


class ScoreCalculator:
    """Analyzes the 5x5 logical grid and computes connected region scores."""

    def _find_connected_region(
        self, grid: List[List], r: int, c: int, visited: List[List[bool]]
    ) -> dict:
        """
        TODO: Breadth-First Search (BFS) for matching terrain.
        Returns dict: {'terrain': str, 'size': int, 'crowns': int}
        """
        pass

    def calculate(self, board_data) -> "ScoreResult":
        """
        TODO:
        1. Init 5x5 False visited matrix.
        2. result = ScoreResult()
        3. Iterate rows, cols:
            If not visited and terrain != Castle:
                region_data = self._find_connected_region(grid, row, col, visited)
                region_score = region_data['size'] * region_data['crowns']
                region_data['score'] = region_score

                result.regions.append(region_data)
                result.total_score += region_score
        4. Return result
        """
        pass
