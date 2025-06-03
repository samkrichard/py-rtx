"""Simple ray tracer implementation used to generate PNG images.

This module defines a minimal set of classes to represent vectors, rays,
geometric primitives, materials and a very small rendering loop that
produces an image using the Pillow library.
"""

from PIL import Image

class Vec3:
    """Simple 3D vector with basic arithmetic helpers."""

    def __init__(self, x: float, y: float, z: float) -> None:
        """Create a new vector from components."""
        self.x = x
        self.y = y
        self.z = z
    
    def dot(self, other: "Vec3") -> float:
        """Return the dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def sqMag(self) -> float:
        """Return the squared magnitude of the vector."""
        return self.x**2 + self.y**2 + self.z**2

    def mag(self) -> float:
        """Return the magnitude of the vector."""
        return self.sqMag()**0.5

    def norm(self) -> "Vec3":
        """Return a normalized copy of the vector."""
        m = self.mag()
        return Vec3(self.x / m, self.y / m, self.z / m)

    def s_mult(self, other: float) -> "Vec3":
        """Return the vector scaled by *other*."""
        return Vec3(self.x * other, self.y * other, self.z * other)

    def add(self, other: "Vec3") -> "Vec3":
        """Return the sum of this vector and *other*."""
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def sub(self, other: "Vec3") -> "Vec3":
        """Return the difference between this vector and *other*."""
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def neg(self) -> "Vec3":
        """Return the negated vector."""
        return Vec3(-self.x, -self.y, -self.z)

def clamp(c: float) -> int:
    """Clamp a color channel value to the 0-255 range."""
    round(c)
    if c > 255:
        c = 255
    elif c < 0:
        c = 0
    return int(c)

class Material:
    """Surface properties for a renderable object."""

    def __init__(self, color, spec, spec_i, reflectivity):
        """Initialize a material.

        Args:
            color: RGB tuple describing the base color.
            spec: Specular exponent.
            spec_i: Specular intensity multiplier.
            reflectivity: Reflection contribution factor.
        """
        self.color = color
        self.spec = spec
        self.spec_i = spec_i
        self.reflectivity = reflectivity
    
class World:
    """Container for scene objects."""

    background_color = (0, 0, 0)

    def __init__(self, bodies=None):
        self.bodies = bodies if bodies else []
        
class Intersection:
    """Stores ray intersection information."""

    def __init__(self, body, t, poi, n):
        self.body = body
        self.t = t
        self.poi = poi
        self.n = n

class Sphere:
    """Simple sphere primitive."""

    def __init__(self, c: Vec3, r: float, m: Material) -> None:
        """Create a sphere.

        Args:
            c: Centre of the sphere.
            r: Radius of the sphere.
            m: Material applied to the surface.
        """
        self.c = c
        self.r = r
        self.m = m

    def intersect(self, ray: "Ray") -> Intersection:
        """Return the intersection of *ray* with this sphere.

        A negative ``t`` value in the returned ``Intersection`` indicates
        that there is no valid intersection in front of the ray origin.
        """
        k = ray.d.dot(ray.o.sub(self.c))
        dis = (k**2) - ((ray.o.sub(self.c)).sqMag() - (self.r**2))
        if dis > 0:
            t1 = -k + dis**0.5
            t2 = -k - dis**0.5
            t = min(t1, t2)
            t = t if t > 0 else max(t1, t2)
            poi = ray.d.s_mult(t)
            return Intersection(self, t, poi, poi.sub(self.c).norm())
        else:
            return Intersection(self, -1, None, None)

class Plane:
    """Infinite plane primitive."""

    def __init__(self, p: Vec3, n: Vec3, m: Material) -> None:
        """Create a plane defined by point ``p`` and normal ``n``."""
        self.p = p
        self.n = n.norm()
        self.m = m

    def intersect(self, ray: "Ray") -> Intersection:
        """Return the intersection of *ray* with this plane."""
        denominator = ray.d.dot(self.n)
        if denominator != 0:
            t = (self.p.sub(ray.o).dot(self.n)) / denominator
            poi = ray.d.s_mult(t)
            return Intersection(self, t, poi, self.n)
        return Intersection(self, -1, None, None)

class Ray:
    """Ray with origin ``o`` and direction ``d``."""

    def __init__(self, o: Vec3, d: Vec3) -> None:
        self.o = o
        self.d = d

class Light:
    """Point light source."""

    def __init__(self, p: Vec3, color) -> None:
        self.p = p
        self.color = color

class Camera:
    """Simple pin-hole camera used for rendering."""

    background = (0, 0, 0)

    def __init__(self, dim: int) -> None:
        """Create a camera with a square output image size ``dim``."""
        self.dim = dim

    def reflection_vector(self, ray: Ray, intersection: Intersection) -> Vec3:
        """Compute the reflection of ``ray`` at the intersection point."""
        return ray.d.sub(intersection.n.s_mult(2 * ray.d.dot(intersection.n))).norm()

    def diffuse(self, intersection: Intersection, ls: Light) -> float:
        """Return diffuse lighting contribution for a point."""
        lv = ls.p.sub(intersection.poi).norm()
        dt = max(0, lv.dot(intersection.n))
        return dt

    def specular(self, ray: Ray, intersection: Intersection, ls: Light) -> float:
        """Return specular lighting contribution for a point."""
        r = self.reflection_vector(ray, intersection)
        sh = (max(0, r.dot(ls.p.sub(intersection.poi).norm()))) ** intersection.body.m.spec
        sh *= intersection.body.m.spec_i
        return sh

    def shadow(self, intersection: Intersection, bodies, ls: Light) -> bool:
        """Check if the intersection point is shadowed from ``ls``."""
        i2l = ls.p.sub(intersection.poi)
        sr_max = i2l.mag()
        sr = Ray(intersection.poi, i2l.norm())

        for body in bodies:
            if body == intersection.body:
                continue
            if 0.005 < body.intersect(sr).t < sr_max - 0.005:
                return True
        return False

    def intersect_ray(self, ray: Ray, bodies) -> Intersection | None:
        """Return the closest intersection of ``ray`` with any body."""
        intersections = []
        for body in bodies:
            i = body.intersect(ray)
            if i.t > 0:
                intersections.append(i)
        if intersections:
            nearest = intersections[0]
            for inter in intersections:
                if inter.t < nearest.t:
                    nearest = inter
            return nearest
        return None

    def reflection(self, ray: Ray, intersection: Intersection, bodies, ls: Light):
        """Compute reflection color contribution."""
        r = self.reflection_vector(ray, intersection)
        i = self.intersect_ray(Ray(intersection.poi, r), bodies)

        if i is not None:
            dt = self.diffuse(i, ls)
            sh = self.specular(Ray(intersection.poi, r), i, ls)
            if self.shadow(i, bodies, ls):
                dt *= 0.1
                sh = 0
            return (
                clamp(dt * (i.body.m.color[0] / 255) * ls.color[0] + sh),
                clamp(dt * (i.body.m.color[1] / 255) * ls.color[1] + sh),
                clamp(dt * (i.body.m.color[2] / 255) * ls.color[2] + sh),
            )
        return Camera.background

    def rtx(self, name: str, bodies, ls: Light) -> None:
        """Render the scene and write the result to ``name``.

        Args:
            name: Base filename for output (``.png`` extension is added).
            bodies: Iterable of scene objects.
            ls: The single light source in the scene.
        """
        im = Image.new("RGB", (self.dim * 2, self.dim))
        out: list[tuple[int, int, int]] = []

        for row in range(self.dim):
            for col in range(self.dim * 2):
                # Generate primary ray direction from pixel coordinates
                o = Vec3(-2 + (2.0 / self.dim) * (col + 0.5),
                         1.0 - (2.0 / self.dim) * (row + 0.5),
                         -1.0)
                ray = Ray(o, o.norm())

                intersection = self.intersect_ray(ray, bodies)

                if intersection is not None:
                    dt = self.diffuse(intersection, ls)
                    sh = self.specular(ray, intersection, ls)
                    reflect = self.reflection(ray, intersection, bodies, ls)

                    if self.shadow(intersection, bodies, ls):
                        dt *= 0.1
                        sh = 0

                    out.append(
                        (
                            clamp(dt * (intersection.body.m.color[0] / 255) * ls.color[0]
                                  + sh * intersection.body.m.color[0]
                                  + reflect[0] * intersection.body.m.reflectivity),
                            clamp(dt * (intersection.body.m.color[1] / 255) * ls.color[1]
                                  + sh * intersection.body.m.color[1]
                                  + reflect[1] * intersection.body.m.reflectivity),
                            clamp(dt * (intersection.body.m.color[2] / 255) * ls.color[2]
                                  + sh * intersection.body.m.color[2]
                                  + reflect[2] * intersection.body.m.reflectivity),
                        )
                    )
                else:
                    out.append(Camera.background)

        im.putdata(out)
        im.save(name + ".png")
    
def main() -> None:
    """Entry point used when running this module as a script."""

    im = Camera(400)

    mat = Material((120, 120, 120), 64, 1, 0.5)
    mat2 = Material((100, 100, 100), 64, 0.7, 0)
    mat3 = Material((50, 200, 100), 64, 1, 0)
    
    sph = Sphere(Vec3(-2, 2, -12), 4, mat)
    sph2 = Sphere(Vec3(-3, 1, -5), 0.5, mat3)
    
    pl = Plane(Vec3(0, -6, 0), Vec3(0, 1, 0), mat2)
    pl2 = Plane(Vec3(0, 0, -70), Vec3(0, 0, 1), mat2)
    
    ls = Light(Vec3(0, 1, 20), (255, 255, 255))

    im.rtx('output1', [sph, sph2, pl, pl2], ls)
    
if __name__ == '__main__':
    main()
