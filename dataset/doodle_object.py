from typing import List, Tuple


class DoodleObject:
    def __init__(self, strokes: List[List[Tuple[int, int]]], cls: str, ordered: bool = True, segments: List[List[str]] = None):
        """
        Parameters
        ----------
        strokes : List[List[Tuple[int, int]]]
            List of strokes, where each stroke is list of points and point is tuple of x and y coordinates.
            x should be first and y should be second element of tuple.

        cls : str
            Class of object

        ordered : bool
            If True, then points in strokes will be ordered by drawing order,
            otherwise unordered points will be set of points (x, y) without any order

        segments : List[List[str]]
            Segment type for each stroke.
            For example, segments for each storke in four-leg animal can be one of the following ['head', 'body', 'legs', 'tail', ...]
        """

        self.strokes = strokes
        self.ordered = ordered
        self.cls = cls
        self.segments = segments

    def __str__(self):
        # Create a string representation of the DoodleObject
        strokes_count = len(self.strokes)
        ordered_status = "Ordered" if self.ordered else "Unordered"

        # Format strokes for better readability
        strokes_info = ', '.join([str(stroke) for stroke in self.strokes])
        segments_info = f", Segments: {self.segments}" if self.segments else ""

        return (f"DoodleObject(class='{self.cls}', "
                f"Strokes Count={strokes_count}, "
                f"Status={ordered_status}, "
                f"Strokes={strokes_info})")


class BBoxInt():

    def __init__(self, points):

        self.x1 = points[0]
        self.y1 = points[1]
        self.x2 = points[2]
        self.y2 = points[3]

    #@property
    def width(self):
        return self.x2 - self.x1

    #@property
    def height(self):
        return self.y2 - self.y1
