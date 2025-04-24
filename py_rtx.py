from PIL import Image

class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def sqMag(self):
        return self.x**2 + self.y**2 + self.z**2

    def mag(self):
        return self.sqMag()**0.5

    def norm(self):
        m = self.mag()
        return Vec3(self.x / m, self.y / m, self.z / m)

    def s_mult(self, other):
        return Vec3(self.x * other, self.y * other, self.z * other)

    def add(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def sub(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def neg(self):
        return Vec3(-self.x, -self.y, -self.z)

def clamp(c):
    round(c)
    if c > 255:
        c = 255
    elif c < 0:
        c = 0
    return int(c)

class Material:
    def __init__(self, color, spec, spec_i, reflectivity):
        self.color = color
        self.spec = spec
        self.spec_i = spec_i
        self.reflectivity = reflectivity
    
class World:
    background_color = (0, 0, 0)
    def __init__(self, bodies=[]):
        self.bodies = bodies
        
class Intersection:
    def __init__(self, body, t, poi, n):
        self.body = body
        self.t = t
        self.poi = poi
        self.n = n

class Sphere:
    def __init__(self, c, r, m):
        self.c = c
        self.r = r
        self.m = m

    def intersect(self, ray):
        k = ray.d.dot(ray.o.sub(self.c))
        dis = (k**2) - ((ray.o.sub(self.c)).sqMag() - (self.r**2))
        if dis > 0:
            t1 = -k + dis**(0.5)
            t2 = -k - dis**(0.5)
            t = min(t1, t2)
            t = t if t > 0 else max(t1, t2)
            poi = ray.d.s_mult(t)
            return Intersection(self, t, poi, poi.sub(self.c).norm())
        else:
            return Intersection(self, -1, None, None)

class Plane:
    def __init__(self, p, n, m):
        self.p = p
        self.n = n.norm()
        self.m = m

    #negative n or no???
    def intersect(self, ray):
        denominator = ray.d.dot(self.n)
        if denominator != 0: 
            t = ((self.p.sub(ray.o)).dot(self.n)) / denominator
            poi = ray.d.s_mult(t)
            return Intersection(self, t, poi, self.n)
        else:
            return Intersection(self, -1, None, None)

class Ray:
    def __init__(self, o, d):
        self.o = o
        self.d = d

class Light:
    def __init__(self, p, color):
        self.p = p
        self.color = color

class Camera:
    
    background = (0, 0, 0)
    
    def __init__(self, dim):
        self.dim = dim

    def reflection_vector(self, ray, intersection):
        return ray.d.sub(intersection.n.s_mult(2 * ray.d.dot(intersection.n))).norm()

    def diffuse(self, intersection, ls):
        lv = (ls.p.sub(intersection.poi)).norm()
        dt = max(0, (lv.dot(intersection.n)))
        return dt

    def specular(self, ray, intersection, ls):
        r = self.reflection_vector(ray, intersection)
        sh = (max(0, r.dot(ls.p.sub(intersection.poi).norm())))**intersection.body.m.spec * intersection.body.m.spec_i
        return sh 

    def shadow(self, intersection, bodies , ls):
        i2l = ls.p.sub(intersection.poi)
        sr_max = i2l.mag()
        sr = Ray(intersection.poi, i2l.norm())

        for body in bodies:
            if body == intersection.body:
                pass
            elif sr_max - 0.005 > body.intersect(sr).t > 0.005:
                return True
        return False

    def intersect_ray(self, ray, bodies):
        intersections = []
        for body in bodies:
            i = body.intersect(ray)
            if i.t > 0:
                intersections += [i]
        if len(intersections) > 0:
            nearest = intersections[0]
            for intersection in intersections:
                if intersection.t < nearest.t:
                    nearest = intersection
            return nearest
        else:
            return None

    def reflection(self, ray, intersection, bodies, ls):
        r = self.reflection_vector(ray, intersection)
        i = self.intersect_ray(Ray(intersection.poi, r), bodies)

        if i != None:
            dt = self.diffuse(i, ls)
            sh = self.specular(Ray(intersection.poi, r), i, ls)
            if self.shadow(i, bodies, ls):
                dt *= 0.1
                sh = 0
            return (
                clamp(dt * (i.body.m.color[0] / 255) * ls.color[0] + sh),
                clamp(dt * (i.body.m.color[1] / 255) * ls.color[1] + sh),
                clamp(dt * (i.body.m.color[2] / 255) * ls.color[2] + sh)
                )
        else:
            return Camera.background

    def rtx(self, name, bodies, ls):
        im = Image.new('RGB', (self.dim * 2, self.dim))
        out = []

        for row in range(self.dim):
            for col in range(self.dim * 2):
                o = Vec3(-2 + (2.0/self.dim) * (col + 0.5), 1.0 - (2.0/self.dim) * (row + 0.5), -1.0)
                ray = Ray(o, o.norm())

                intersection = self.intersect_ray(ray, bodies)

                if intersection != None:
                    dt = self.diffuse(intersection, ls)

                    sh = self.specular(ray, intersection, ls)

                    reflect = self.reflection(ray, intersection, bodies, ls)

                    if (self.shadow(intersection, bodies, ls)):
                        dt *= 0.1
                        sh = 0
                    
                    out += [(
                        clamp(dt * (intersection.body.m.color[0] / 255) * ls.color[0] + sh * intersection.body.m.color[0] + reflect[0] * intersection.body.m.reflectivity),
                        clamp(dt * (intersection.body.m.color[1] / 255) * ls.color[1] + sh * intersection.body.m.color[1] + reflect[1] * intersection.body.m.reflectivity),
                        clamp(dt * (intersection.body.m.color[2] / 255) * ls.color[2] + sh * intersection.body.m.color[2] + reflect[2] * intersection.body.m.reflectivity)
                        )]

                else:
                    out += [Camera.background]
                    
        im.putdata(out)
        im.save(name + '.png')
    
def main():
    im = Camera(400)

    mat = Material((120, 120, 120), 64, 1, 0.5)
    mat2 = Material((100, 100, 100), 64, 0.7, 0)
    mat3 = Material((50, 200, 100), 64, 1, 0)
    
    sph = Sphere(Vec3(-2, 2, -12), 4, mat)
    sph2 = Sphere(Vec3(-3, 1, -5), 0.5, mat3)
    
    pl = Plane(Vec3(0, -6, 0), Vec3(0, 1, 0), mat2)
    pl2 = Plane(Vec3(0, 0, -70), Vec3(0, 0, 1), mat2)
    
    ls = Light(Vec3(0, 1, 20), (255, 255, 255))

    im.rtx('new_test2025_2', [sph, sph2, pl, pl2], ls)
    
if __name__ == '__main__':
    main()
