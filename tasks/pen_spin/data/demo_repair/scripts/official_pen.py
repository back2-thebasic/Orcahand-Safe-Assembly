"""Attach Wuji's official 250 mm pen without inheriting Orca geom defaults."""
from pathlib import Path
import copy
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

ASSET = Path(__file__).resolve().parents[3]/'models/pen/assets/pen_tracking.xml'


def attach_official_pen(scene_tree, mass=.030):
    source = mujoco.MjModel.from_xml_path(str(ASSET))
    root = ET.parse(ASSET).getroot()
    target = scene_tree.getroot()
    asset = target.find('asset')
    texture_dir = (ASSET.parent/'../textures').resolve()
    for element in root.find('asset'):
        item = copy.deepcopy(element)
        for key in ('name','texture'):
            if key in item.attrib:item.set(key,'wuji_pen_'+item.get(key))
        if item.tag=='texture':
            for key,value in list(item.attrib.items()):
                if key.startswith('file'):item.set(key,str(texture_dir/value))
        asset.append(item)
    pen = copy.deepcopy(root.find('worldbody/body'))
    pen.set('name','pen')
    pen.find('freejoint').set('name','pen_free')
    pen.find('site').set('name','wuji_pen_center')
    sid=source.body('cube').id
    expected_inertia=source.body_inertia[sid]*(mass/source.body_mass[sid])
    def nums(x):return ' '.join(str(float(v)) for v in x)
    ET.SubElement(pen,'inertial',pos=nums(source.body_ipos[sid]),
        quat=nums(source.body_iquat[sid]),mass=str(mass),diaginertia=nums(expected_inertia))
    # A dedicated class prevents hand-specific density, margin, solref, etc. leaking in.
    default=target.find('default')
    if default is None:default=ET.SubElement(target,'default')
    dedicated=ET.SubElement(default,'default',{'class':'wuji_pen_defaults'})
    ET.SubElement(dedicated,'geom',type='sphere',size='.01',rgba='1 1 1 1',
        friction='1 .005 .0001',condim='3',margin='0',gap='0',priority='0',
        solref='.02 1',solimp='.9 .95 .001 .5 2')
    for g in pen.findall('geom'):
        g.set('class','wuji_pen_defaults')
        g.set('material','wuji_pen_'+g.get('material'))
        g.set('name','wuji_pen_'+g.get('name'))
    world=target.find('worldbody')
    world.remove(world.find("body[@name='pen']"))
    world.append(pen)
    return source, expected_inertia


def verify_official_pen(model, source, inertia):
    body=model.body('pen').id
    np.testing.assert_allclose(model.body_mass[body],.030,atol=1e-12)
    np.testing.assert_allclose(model.body_inertia[body],inertia,atol=1e-12)
    np.testing.assert_allclose(model.body_ipos[body],source.body_ipos[source.body('cube').id],atol=1e-12)
    for name in ('cube','A_bottom','A_top'):
        a=model.geom('wuji_pen_'+name).id;b=source.geom(name).id
        for field in ('geom_type','geom_size','geom_pos','geom_quat','geom_friction',
                      'geom_condim','geom_contype','geom_conaffinity','geom_solref','geom_solimp','geom_margin','geom_gap'):
            np.testing.assert_allclose(getattr(model,field)[a],getattr(source,field)[b],atol=1e-12,err_msg=field)
    return dict(mass_kg=float(model.body_mass[body]),principal_inertia_kg_m2=model.body_inertia[body].tolist(),
        center_of_mass_local_m=model.body_ipos[body].tolist(),length_m=.250,shaft_diameter_m=.0149,
        collision_geoms=3,freejoint=True,source_properties_verified=True)
