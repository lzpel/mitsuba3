"""Gradients of point-light position and directional-light direction.

The smooth part of these derivatives (cosine / BSDF term and 1/r^2 falloff) is
obtained by plain AD; these tests check it against central finite differences on
an unoccluded scene (no shadow boundary, so the FD and AD must agree).
"""
import pytest
import drjit as dr
import mitsuba as mi


def make_scene(integrator, light):
    T = mi.ScalarTransform4f()
    d = {
        'type': 'scene',
        'integrator': {'type': integrator} if integrator == 'direct' else {'type': integrator, 'max_depth': 2},
        'sensor': {
            'type': 'perspective', 'fov': 40,
            'to_world': T.look_at([0, -6, 3], [0, 0, 0], [0, 0, 1]),
            'film': {'type': 'hdrfilm', 'width': 16, 'height': 16, 'pixel_filter': {'type': 'box'}},
        },
        'floor': {'type': 'rectangle', 'to_world': T.scale(3), 'bsdf': {'type': 'diffuse'}},
    }
    if light == 'point':
        d['light'] = {'type': 'point', 'position': [0.5, 0.3, 3.0], 'intensity': {'type': 'rgb', 'value': 20.0}}
    else:
        d['light'] = {'type': 'directional', 'direction': [0.3, 0.2, -1.0], 'irradiance': {'type': 'rgb', 'value': 2.0}}
    return mi.load_dict(d)


def loss(scene, params, spp=256):
    return dr.sum(mi.render(scene, params, spp=spp, seed=1, seed_grad=2, spp_grad=spp).array)


@pytest.mark.parametrize('integrator', ['direct', 'prb', 'direct_projective', 'prb_projective'])
def test01_point_position_grad(variants_all_ad_rgb, integrator):
    scene = make_scene(integrator, 'point')
    params = mi.traverse(scene)
    key = 'light.position'
    assert (params.flags(key) & mi.ParamFlags.NonDifferentiable) == 0
    assert (params.flags(key) & mi.ParamFlags.Discontinuous) != 0

    p0 = mi.Point3f(params[key])
    eps = 1e-2
    params[key] = p0 + mi.Point3f(eps, 0, 0); params.update()
    lp = loss(scene, params)[0]
    params[key] = p0 - mi.Point3f(eps, 0, 0); params.update()
    lm = loss(scene, params)[0]
    fd = (lp - lm) / (2 * eps)

    params[key] = p0; params.update()
    dr.enable_grad(params[key]); params.update()
    L = loss(scene, params)
    dr.backward(L)
    g = dr.grad(params[key])
    assert dr.allclose(g.x, fd, rtol=0.05), (g, fd)
    assert dr.abs(g.y)[0] > 0 and dr.abs(g.z)[0] > 0


@pytest.mark.parametrize('integrator', ['direct', 'prb', 'direct_projective', 'prb_projective'])
def test02_directional_direction_grad(variants_all_ad_rgb, integrator):
    scene = make_scene(integrator, 'directional')
    params = mi.traverse(scene)
    key = 'light.direction'
    assert key in params.keys()
    assert (params.flags(key) & mi.ParamFlags.NonDifferentiable) == 0

    d0 = mi.Vector3f(params[key])
    delta = dr.normalize(dr.cross(mi.Vector3f(1, 0, 0), dr.normalize(d0)))
    eps = 1e-2
    params[key] = d0 + eps * delta; params.update()
    lp = loss(scene, params)[0]
    params[key] = d0 - eps * delta; params.update()
    lm = loss(scene, params)[0]
    fd = (lp - lm) / (2 * eps)

    t = mi.Float(0.0)
    dr.enable_grad(t)
    params[key] = d0 + t * delta; params.update()
    L = loss(scene, params)
    dr.backward(L)
    g = dr.grad(t)[0]
    assert dr.allclose(g, fd, rtol=0.05), (g, fd)


def test03_directional_to_world_sync(variants_all_ad_rgb):
    scene = make_scene('direct', 'directional')
    params = mi.traverse(scene)
    params['light.direction'] = mi.Vector3f(0, 0, -2)   # not normalized on purpose
    params.update()
    tw = mi.AffineTransform4f(params['light.to_world'])
    d = tw @ mi.Vector3f(0, 0, 1)
    assert dr.allclose(d, mi.Vector3f(0, 0, -1), atol=1e-5)
