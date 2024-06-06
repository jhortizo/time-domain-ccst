"""
Create meshes programmatically, using gmsh API for python
"""

import gmsh


def _create_square_mesh(side: float, mesh_size: float, mesh_file: str) -> None:
    coords = [(0, 0), (side, 0), (side, side), (0, side)]
    _create_mesh_from_coords(coords, mesh_size, mesh_file)


def _create_rectangle_mesh(
    side_x: float, side_y: float, mesh_size: float, mesh_file: str
) -> None:
    coords = [(0, 0), (side_x, 0), (side_x, side_y), (0, side_y)]
    _create_mesh_from_coords(coords, mesh_size, mesh_file)


def _create_triangle_mesh(cathetus: float, mesh_size: float, mesh_file: str):
    coords = [(0, 0), (cathetus, 0), (0, cathetus)]
    _create_mesh_from_coords(coords, mesh_size, mesh_file)


def _create_quarter_ring_mesh(
    inner_radius: float, outer_radius: float, mesh_size: float, mesh_file: str
):
    coords = [
        (inner_radius, 0),
        (outer_radius, 0),
        (0, outer_radius),
        (0, inner_radius),
    ]

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("custom")

    # Ensure we are using quadrilateral elements
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
    gmsh.option.setNumber("Mesh.ElementOrder", 2)  # Quadratic elements
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # 1: All quadrangles

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    lc = mesh_size
    center = gmsh.model.geo.addPoint(0, 0, 0, lc)

    ps = []
    for coord in coords:
        ps.append(gmsh.model.geo.addPoint(coord[0], coord[1], 0, lc))

    lines = []
    lines.append(gmsh.model.geo.addLine(ps[0], ps[1]))
    lines.append(gmsh.model.geo.addCircleArc(ps[1], center, ps[2]))
    lines.append(gmsh.model.geo.addLine(ps[2], ps[3]))
    lines.append(gmsh.model.geo.addCircleArc(ps[3], center, ps[0]))

    cl = gmsh.model.geo.addCurveLoop(lines)
    pl = gmsh.model.geo.addPlaneSurface([cl])

    for i, line in enumerate(lines):
        gmsh.model.geo.addPhysicalGroup(1, [line], i + 1)
        gmsh.model.setPhysicalName(1, i + 1, f"Edge {i + 1}")

    gmsh.model.geo.addPhysicalGroup(2, [pl], 1)
    gmsh.model.setPhysicalName(2, 1, "Surface")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    gmsh.write(mesh_file)

    # dev code, add breakpoint and check mesh
    # gmsh.fltk.run()
    gmsh.finalize()


def _create_plate_hole_mesh(
    hole_radius: float, side: float, mesh_size: float, mesh_file: str
):
    coords = [
        (hole_radius, 0),
        (hole_radius + side, 0),
        (hole_radius + side, hole_radius + side),
        (0, hole_radius + side),
        (0, hole_radius),
    ]

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("custom")

    # Ensure we are using quadrilateral elements
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
    gmsh.option.setNumber("Mesh.ElementOrder", 2)  # Quadratic elements
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # 1: All quadrangles

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    lc = mesh_size
    center = gmsh.model.geo.addPoint(0, 0, 0, lc)

    ps = []
    for coord in coords:
        ps.append(gmsh.model.geo.addPoint(coord[0], coord[1], 0, lc))

    lines = []
    lines.append(gmsh.model.geo.addLine(ps[0], ps[1]))
    lines.append(gmsh.model.geo.addLine(ps[1], ps[2]))
    lines.append(gmsh.model.geo.addLine(ps[2], ps[3]))
    lines.append(gmsh.model.geo.addLine(ps[3], ps[4]))
    lines.append(gmsh.model.geo.addCircleArc(ps[4], center, ps[0]))

    cl = gmsh.model.geo.addCurveLoop(lines)
    pl = gmsh.model.geo.addPlaneSurface([cl])

    for i, line in enumerate(lines):
        gmsh.model.geo.addPhysicalGroup(1, [line], i + 1)
        gmsh.model.setPhysicalName(1, i + 1, f"Edge {i + 1}")

    gmsh.model.geo.addPhysicalGroup(2, [pl], 1)
    gmsh.model.setPhysicalName(2, 1, "Surface")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    gmsh.write(mesh_file)

    # dev code, add breakpoint and check mesh
    # gmsh.fltk.run()
    gmsh.finalize()


def _create_circle_mesh(
    radius: float, mesh_size: float, mesh_file: str
):
    coords = [
        (radius, 0),
        (-radius, 0),
    ]

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("custom")

    # Ensure we are using quadrilateral elements
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
    gmsh.option.setNumber("Mesh.ElementOrder", 2)  # Quadratic elements
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # 1: All quadrangles

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    lc = mesh_size
    center = gmsh.model.geo.addPoint(0, 0, 0, lc)

    ps = []
    for coord in coords:
        ps.append(gmsh.model.geo.addPoint(coord[0], coord[1], 0, lc))

    lines = []
    lines.append(gmsh.model.geo.addCircleArc(ps[0], center, ps[1]))
    lines.append(gmsh.model.geo.addCircleArc(ps[1], center, ps[0]))

    cl = gmsh.model.geo.addCurveLoop(lines)
    pl = gmsh.model.geo.addPlaneSurface([cl])

    for i, line in enumerate(lines):
        gmsh.model.geo.addPhysicalGroup(1, [line], i + 1)
        gmsh.model.setPhysicalName(1, i + 1, f"Edge {i + 1}")

    gmsh.model.geo.addPhysicalGroup(2, [pl], 1)
    gmsh.model.setPhysicalName(2, 1, "Surface")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    gmsh.write(mesh_file)

    # dev code, add breakpoint and check mesh
    # gmsh.fltk.run()
    gmsh.finalize()


def _create_mesh_from_coords(
    coords: list[tuple], mesh_size: float, mesh_file: str
) -> None:
    """Create a mesh by connectiong the given set of coordinates with straight lines"""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("custom")

    # Ensure we are using quadrilateral elements
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
    gmsh.option.setNumber("Mesh.ElementOrder", 2)  # Quadratic elements
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # 1: All quadrangles

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    lc = mesh_size
    ps = []
    for coord in coords:
        ps.append(gmsh.model.geo.addPoint(coord[0], coord[1], 0, lc))

    lines = []
    for i in range(len(ps)):
        lines.append(gmsh.model.geo.addLine(ps[i], ps[(i + 1) % len(ps)]))

    cl = gmsh.model.geo.addCurveLoop(lines)
    pl = gmsh.model.geo.addPlaneSurface([cl])

    for i, line in enumerate(lines):
        gmsh.model.geo.addPhysicalGroup(1, [line], i + 1)
        gmsh.model.setPhysicalName(1, i + 1, f"Edge {i + 1}")

    gmsh.model.geo.addPhysicalGroup(2, [pl], 1)
    gmsh.model.setPhysicalName(2, 1, "Surface")

    gmsh.model.geo.mesh.setRecombine(2, pl)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    gmsh.write(mesh_file)

    # dev code, add breakpoint and check mesh
    # gmsh.fltk.run()
    gmsh.finalize()


def create_mesh(geometry_type: str, params: dict, mesh_file: str) -> None:
    mesh_functions = {
        "square": _create_square_mesh,
        "triangle": _create_triangle_mesh,
        "quarter_ring": _create_quarter_ring_mesh,
        "rectangle": _create_rectangle_mesh,
        "plate_hole": _create_plate_hole_mesh,
        "circle": _create_circle_mesh,
    }
    if geometry_type in mesh_functions:
        mesh_functions[geometry_type](**params, mesh_file=mesh_file)
    else:
        raise ValueError(f"Unknown geometry type: {geometry_type}")
