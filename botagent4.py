"""
Spider Rig Builder - All-in-One Version
========================================
Builds a complete quadruped spider rig with:
- Modular leg system with IK
- Circular walk paths with gait offsets
- Steering controller
- Organic body noise
- Toggleable body shape (cube/sphere)
- Random walk behavior (toggleable)
- Ground/terrain following (toggleable)

This version includes all necessary modules inline to avoid external dependencies.

Controls on character_controller:
- body_shape: 0 = Cube, 1 = Sphere
- random_walk: 0 = Off, 1 = On
- walk_speed: Movement speed (default 0.05)
- turn_speed: Turning speed (default 0.02)
- ground_follow: 0 = Off, 1 = On
- body_height: Height above ground (default 3.0)

Ground Detection:
- Create a mesh named "Ground" for terrain following
- The ground can be flat, bumpy, curved, or any shape
- Spider will walk over the surface following its contours
"""

import bpy
import math
import random
import sys
import traceback
from mathutils import Vector, Matrix, noise
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# =============================================================================
# EMBEDDED MODULE: create_spider_assembly
# =============================================================================

def create_spider_assembly():
    import bmesh
    
    # --------------------------------------------------------------------
    # 1. CLEANUP
    # --------------------------------------------------------------------
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for _ in range(3):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

    # --------------------------------------------------------------------
    # 2. CREATE BODY (Cube)
    # --------------------------------------------------------------------
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 3.0))
    body = bpy.context.active_object
    body.name = "Spider_Body"
    
    print("Created Body at Z=3.0")

    # --------------------------------------------------------------------
    # 3. CREATE LEG MESH
    # --------------------------------------------------------------------
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    leg_mesh = bpy.context.active_object
    leg_mesh.name = "Leg_Mesh"
    
    leg_mesh.rotation_euler.x = math.radians(180)
    leg_mesh.location = Vector((0, -0.6, 3.0))
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.merge(type='CENTER')
    bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":(0, 0, 4), "orient_type":'LOCAL'})
    
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.subdivide(number_cuts=2)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    mod = leg_mesh.modifiers.new(name="Skin", type='SKIN')
    
    bpy.ops.object.mode_set(mode='EDIT')
    me = leg_mesh.data
    bm = bmesh.from_edit_mesh(me)
    skin_layer = bm.verts.layers.skin.verify()
    
    for v in bm.verts:
        z = v.co.z
        skin_data = v[skin_layer]
        radius = 0.5 - (z * 0.075)
        if radius < 0.1: radius = 0.1
        skin_data.radius = (radius, radius)
        
    bmesh.update_edit_mesh(me)

    # --------------------------------------------------------------------
    # 4. GENERATE ARMATURE
    # --------------------------------------------------------------------
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = leg_mesh
    bpy.ops.object.skin_armature_create(modifier="Skin")
    
    armature = bpy.context.active_object
    armature.name = "Leg_Rig"
    
    # --------------------------------------------------------------------
    # 5. IK TARGET
    # --------------------------------------------------------------------
    target_loc = Vector((0, -2.4562, 0.28973))
    
    bpy.ops.object.empty_add(type='SPHERE', radius=0.4, location=target_loc)
    ik_target = bpy.context.active_object
    ik_target.name = "IK_Target"
    
    # --------------------------------------------------------------------
    # 6. APPLY IK
    # --------------------------------------------------------------------
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    target_bone = None
    dist_max = -1.0
    
    for pbone in armature.pose.bones:
        head_dist = pbone.head.length
        tail_dist = pbone.tail.length
        if tail_dist > dist_max:
            dist_max = tail_dist
            target_bone = pbone
            
    if target_bone:
        c = target_bone.constraints.new('IK')
        c.target = ik_target
        c.chain_count = 0
        print(f"IK Applied to bone: {target_bone.name}")
        
        parent_bone = target_bone.parent
        if parent_bone:
             parent_bone.rotation_mode = 'XYZ'
             parent_bone.rotation_euler.x = math.radians(45)
             print(f"Applied rotation hint to {parent_bone.name}")
    else:
        print("Error: Could not find tip bone.")

    # --------------------------------------------------------------------
    # 7. PARENT LEG TO BODY
    # --------------------------------------------------------------------
    bpy.ops.object.mode_set(mode='OBJECT')
    
    leg_mesh.parent = body
    leg_mesh.matrix_parent_inverse = body.matrix_world.inverted()
    
    armature.parent = body
    armature.matrix_parent_inverse = body.matrix_world.inverted()
    
    print("Parented Leg Mesh and Armature to Body.")


# =============================================================================
# EMBEDDED MODULE: create_path
# =============================================================================

def create_circular_path():
    if "WalkPath" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["WalkPath"], do_unlink=True)

    target_loc = Vector((0, -2.4562, 0.28973))
    
    curve_data = bpy.data.curves.new(name="WalkPathData", type='CURVE')
    curve_data.dimensions = '3D'
    
    spline = curve_data.splines.new('BEZIER')
    spline.use_cyclic_u = True
    
    r = 0.8
    h = 1.2 
    
    spline.bezier_points.add(2)
    
    p0 = spline.bezier_points[0]
    p0.co = Vector((-r, 0, 0))
    p0.handle_left = Vector((-r, 0, 0))
    p0.handle_right = Vector((-r, 0, r))
    p0.handle_left_type = 'VECTOR'
    p0.handle_right_type = 'FREE'
    
    p1 = spline.bezier_points[1]
    p1.co = Vector((0, 0, h))
    p1.handle_left = Vector((-r*0.5, 0, h)) 
    p1.handle_right = Vector((r*0.5, 0, h))
    p1.handle_left_type = 'ALIGNED'
    p1.handle_right_type = 'ALIGNED'

    p2 = spline.bezier_points[2]
    p2.co = Vector((r, 0, 0))
    p2.handle_left = Vector((r, 0, r))
    p2.handle_right = Vector((r, 0, 0))
    p2.handle_left_type = 'FREE'
    p2.handle_right_type = 'VECTOR'
    
    object_loc = Vector((target_loc.x, target_loc.y, 0.0))
    
    path = bpy.data.objects.new("WalkPath", curve_data)
    path.location = object_loc
    path.rotation_euler.z = math.radians(90)
    
    bpy.context.collection.objects.link(path)
    bpy.context.view_layer.objects.active = path
    
    print(f"Created Custom 'WalkPath' (D-Shape) at {object_loc}, Rotated 90 Z.")

    if "IK_Target" in bpy.data.objects:
        ik_target = bpy.data.objects["IK_Target"]
        
        to_remove = [c for c in ik_target.constraints if c.type == 'FOLLOW_PATH']
        for c in to_remove:
            ik_target.constraints.remove(c)
            
        if ik_target.animation_data:
            ik_target.animation_data_clear()
        
        c = ik_target.constraints.new('FOLLOW_PATH')
        c.name = "WalkConstraint"
        c.target = path
        
        ik_target.location = (0, 0, 0)
        
        c.offset = 0
        c.keyframe_insert(data_path="offset", frame=10)
        
        c.offset = 100
        c.keyframe_insert(data_path="offset", frame=30)
        
        if ik_target.animation_data and ik_target.animation_data.action:
            action = ik_target.animation_data.action
            
            def recursive_fix(obj, visited=None):
                if visited is None: visited = set()
                if obj in visited: return
                visited.add(obj)
                
                if hasattr(obj, 'data_path') and hasattr(obj, 'keyframe_points') and hasattr(obj, 'extrapolation'):
                     if 'WalkConstraint' in obj.data_path and 'offset' in obj.data_path:
                        obj.extrapolation = 'LINEAR'
                        for k in obj.keyframe_points:
                            k.interpolation = 'LINEAR'
                        print(f"SUCCESS: Applied LINEAR Extrapolation to: {obj.data_path}")
                     return

                search_attrs = ['layers', 'strips', 'channelbags', 'fcurves']
                for attr in search_attrs:
                    if hasattr(obj, attr):
                        try:
                            for item in getattr(obj, attr):
                                recursive_fix(item, visited)
                        except: pass

            print("Searching F-Curves recursively...")
            recursive_fix(action)

        print("Animated Follow Path Offset (Frame 10->30, 0->100, Linear Loop).")
    else:
        print("Warning: IK_Target object not found. Created path but did not attach.")


# =============================================================================
# EMBEDDED MODULE: create_body_noise
# =============================================================================

def add_body_noise(target_name="Spider_Body"):
    """Add noise modifiers to F-Curves for organic movement."""
    print(f"    Adding noise to: {target_name}")

    if target_name not in bpy.data.objects:
        print(f"    Warning: {target_name} not found, skipping noise")
        return
        
    body = bpy.data.objects[target_name]
    
    # Insert keyframes for location and rotation (creates F-Curves)
    body.keyframe_insert(data_path="location", frame=1)
    body.keyframe_insert(data_path="location", frame=100)
    body.keyframe_insert(data_path="rotation_euler", frame=1)
    body.keyframe_insert(data_path="rotation_euler", frame=100)
    
    if not body.animation_data or not body.animation_data.action:
        print(f"    Warning: Could not create animation data for {target_name}")
        return
    
    action = body.animation_data.action
    
    # Collect all F-Curves (handles Blender 4.x layered animation system)
    all_fcurves = []
    
    def recursive_find_fcurves(obj, visited=None):
        """Recursively search for F-Curves in Blender's animation data structure."""
        if visited is None:
            visited = set()
        if id(obj) in visited:
            return
        visited.add(id(obj))
        
        # Direct fcurves attribute
        if hasattr(obj, 'fcurves'):
            try:
                for fc in obj.fcurves:
                    if fc not in all_fcurves:
                        all_fcurves.append(fc)
            except:
                pass
        
        # Blender 4.x uses layers -> strips -> channelbags -> fcurves
        search_attrs = ['layers', 'strips', 'channelbags']
        for attr in search_attrs:
            if hasattr(obj, attr):
                try:
                    for item in getattr(obj, attr):
                        recursive_find_fcurves(item, visited)
                except:
                    pass

    recursive_find_fcurves(action)
    
    print(f"    Found {len(all_fcurves)} F-Curves on '{target_name}'")

    if not all_fcurves:
        print(f"    Warning: No F-Curves found on {target_name}")
        return
    
    # Add noise modifier to each F-Curve
    noise_count = 0
    for fcurve in all_fcurves:
        # Clear existing noise modifiers
        to_remove = [m for m in fcurve.modifiers if m.type == 'NOISE']
        for m in to_remove:
            fcurve.modifiers.remove(m)
        
        # Add Noise Modifier
        noise = fcurve.modifiers.new('NOISE')
        
        noise.scale = 15.0
        noise.phase = random.uniform(0, 100)
        noise.use_restricted_range = False
        
        # Different strength for location vs rotation
        if "location" in fcurve.data_path:
            noise.strength = 1.0
        elif "rotation" in fcurve.data_path:
            noise.strength = 0.4
        
        noise.show_expanded = False
        noise_count += 1

    print(f"    Applied noise to {noise_count} F-Curves on '{target_name}'")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SpiderConfig:
    """Central configuration for the spider rig."""
    
    # Naming
    body_name: str = "Spider_Body"
    leg_prefix: str = "Leg"
    ik_target_prefix: str = "IK_Target"
    walk_path_prefix: str = "WalkPath"
    direction_controller_name: str = "Direction_Controller"
    master_controller_name: str = "character_controller"
    
    # Body shape options
    body_sphere_name: str = "Spider_Body_Sphere"
    
    # Ground detection
    ground_plane_name: str = "Ground"
    
    # Component tagging
    component_key: str = "spider_component"
    leg_tag: str = "leg"
    body_tag: str = "body"
    controller_tag: str = "controller"
    
    # Leg configuration
    num_legs: int = 4
    leg_angles: List[float] = field(default_factory=lambda: [0, 90, 180, 270])
    
    # Gait configuration
    gait_offset_frames: float = -10.0
    offset_leg_indices: List[int] = field(default_factory=lambda: [1, 3])  # Alternating gait
    
    # Display
    controller_display_size: float = 3.0


# =============================================================================
# UTILITIES
# =============================================================================

class BlenderContext:
    """Utility methods for Blender operations."""
    
    @staticmethod
    @contextmanager
    def mode(target_mode: str):
        """Context manager for temporarily switching modes."""
        previous_mode = bpy.context.mode
        if previous_mode != target_mode:
            try:
                bpy.ops.object.mode_set(mode=target_mode)
            except RuntimeError:
                pass
        try:
            yield
        finally:
            if previous_mode != target_mode:
                try:
                    bpy.ops.object.mode_set(mode=previous_mode)
                except RuntimeError:
                    pass
    
    @staticmethod
    def deselect_all():
        """Deselect all objects safely."""
        try:
            bpy.ops.object.select_all(action='DESELECT')
        except RuntimeError:
            for obj in bpy.data.objects:
                obj.select_set(False)
    
    @staticmethod
    def select_objects(objects: List[bpy.types.Object], active: Optional[bpy.types.Object] = None):
        """Select specified objects and optionally set active."""
        BlenderContext.deselect_all()
        for obj in objects:
            obj.select_set(True)
        if active:
            bpy.context.view_layer.objects.active = active
        elif objects:
            bpy.context.view_layer.objects.active = objects[0]


# =============================================================================
# CORE COMPONENTS
# =============================================================================

class SpiderRigBuilder:
    """Main builder class for the spider rig."""
    
    def __init__(self, config: Optional[SpiderConfig] = None):
        self.config = config or SpiderConfig()
        self.leg_objects: Dict[int, List[bpy.types.Object]] = {}
        
    def build(self, clear_scene: bool = True) -> bool:
        """Execute the full spider rig build."""
        print("=" * 60)
        print("SPIDER RIG BUILDER (ALL-IN-ONE) - Starting Build")
        print("=" * 60)
        
        try:
            if clear_scene:
                self._clear_scene()
            
            steps = [
                ("Creating Base Assembly", self._create_base_assembly),
                ("Creating Walk Path", self._create_walk_path),
                ("Duplicating Legs", self._duplicate_legs),
                ("Applying Gait Offsets", self._apply_gait_offsets),
                ("Creating Steering Controller", self._create_steering_controller),
                ("Creating Master Controller", self._create_master_controller),
                ("Setting Up Body Shape Toggle", self._setup_body_shape_toggle),
                ("Setting Up Random Walk", self._setup_random_walk),
                ("Setting Up Ground Detection", self._setup_ground_detection),
                ("Adding Body Noise", self._add_body_noise),
            ]
            
            for i, (description, step_func) in enumerate(steps, 1):
                print(f"\n[{i}/{len(steps)}] {description}...")
                try:
                    step_func()
                    print(f"    ✓ Complete")
                except Exception as step_error:
                    print(f"    ✗ STEP FAILED: {step_error}")
                    traceback.print_exc()
                    continue
            
            print("\n" + "=" * 60)
            print("SPIDER RIG BUILD COMPLETE!")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\n✗ BUILD FAILED: {e}")
            traceback.print_exc()
            return False
    
    # -------------------------------------------------------------------------
    # Build Steps
    # -------------------------------------------------------------------------
    
    def _clear_scene(self):
        """Clear all objects from the scene."""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        self.leg_objects.clear()
    
    def _create_base_assembly(self):
        """Create the spider body and first leg."""
        create_spider_assembly()
        self._tag_initial_components()
    
    def _create_walk_path(self):
        """Create the circular walk path for the first leg."""
        create_circular_path()
        self._tag_new_objects_as_leg()
    
    def _tag_new_objects_as_leg(self):
        """Tag any untagged objects (except body) as leg components."""
        body = bpy.data.objects.get(self.config.body_name)
        for obj in bpy.data.objects:
            if obj != body and self.config.component_key not in obj:
                obj[self.config.component_key] = self.config.leg_tag
    
    def _duplicate_legs(self):
        """Duplicate the leg assembly (including walkpath) for all remaining legs."""
        body = bpy.data.objects.get(self.config.body_name)
        base_leg_objects = [o for o in bpy.data.objects if o != body]
        
        if not base_leg_objects:
            raise RuntimeError("No leg objects found to duplicate")
        
        print(f"    Base objects to duplicate: {[o.name for o in base_leg_objects]}")
        
        self.leg_objects[0] = base_leg_objects.copy()
        
        for leg_index in range(1, self.config.num_legs):
            angle = self.config.leg_angles[leg_index]
            new_objects = self._duplicate_and_rotate(base_leg_objects, angle, leg_index)
            self.leg_objects[leg_index] = new_objects
            print(f"    Leg {leg_index} objects: {[o.name for o in new_objects]}")
        
        self._reset_path_rotations()
        self._reassign_ik_targets_to_paths()
    
    def _apply_gait_offsets(self):
        """Apply frame offsets to create alternating gait pattern."""
        offsets_applied = 0
        for leg_index in self.config.offset_leg_indices:
            ik_target = self._get_leg_ik_target(leg_index)
            if ik_target and ik_target.animation_data and ik_target.animation_data.action:
                self._shift_animation(
                    ik_target.animation_data.action,
                    self.config.gait_offset_frames
                )
                offsets_applied += 1
                print(f"    Offset applied to leg {leg_index} ({ik_target.name})")
        
        if offsets_applied == 0:
            print("    No animation data found to offset (this may be OK)")
    
    def _create_steering_controller(self):
        """Create the direction controller and link paths via Copy Rotation."""
        BlenderContext.deselect_all()
        
        ctrl = bpy.data.objects.get(self.config.direction_controller_name)
        
        if not ctrl:
            bpy.ops.object.empty_add(type='SPHERE', location=(0, 0, 0))
            ctrl = bpy.context.active_object
            ctrl.name = self.config.direction_controller_name
            ctrl.empty_display_size = self.config.controller_display_size
            ctrl[self.config.component_key] = self.config.controller_tag
            print(f"    Created '{self.config.direction_controller_name}' (Sphere Empty)")
        else:
            print(f"    '{self.config.direction_controller_name}' already exists")
        
        paths_constrained = 0
        for obj in bpy.data.objects:
            if obj.name.startswith(self.config.walk_path_prefix):
                self._add_constraint_if_missing(obj, 'COPY_ROTATION', ctrl, "Steering")
                paths_constrained += 1
        
        print(f"    Linked {paths_constrained} walk paths to direction controller")
    
    def _create_master_controller(self):
        """Create the master 'character_controller' and parent the rig hierarchy."""
        BlenderContext.deselect_all()
        
        master_name = self.config.master_controller_name
        
        if master_name in bpy.data.objects:
            master = bpy.data.objects[master_name]
            print(f"    '{master_name}' already exists, using existing")
        else:
            bpy.ops.object.empty_add(type='CUBE', location=(0, 0, 0))
            master = bpy.context.active_object
            master.name = master_name
            master.empty_display_size = 5.0
            master[self.config.component_key] = self.config.controller_tag
            print(f"    Created '{master_name}' (Cube Empty)")
        
        children_to_parent = []
        
        dir_ctrl = bpy.data.objects.get(self.config.direction_controller_name)
        if dir_ctrl:
            children_to_parent.append(dir_ctrl)
        
        body = bpy.data.objects.get(self.config.body_name)
        if body:
            children_to_parent.append(body)
        
        for obj in bpy.data.objects:
            if obj.name.startswith(self.config.walk_path_prefix):
                children_to_parent.append(obj)
        
        if not children_to_parent:
            print("    Warning: No children found to parent!")
            return
        
        BlenderContext.deselect_all()
        for child in children_to_parent:
            child.select_set(True)
        
        master.select_set(True)
        bpy.context.view_layer.objects.active = master
        bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
        
        print(f"    Parented {len(children_to_parent)} objects to '{master_name}'")
    
    def _setup_body_shape_toggle(self):
        """Create toggleable body shapes (original/sphere)."""
        BlenderContext.deselect_all()
        
        master = bpy.data.objects.get(self.config.master_controller_name)
        original_body = bpy.data.objects.get(self.config.body_name)
        
        if not original_body or not master:
            print("    Warning: Missing body or master controller, skipping shape toggle")
            return
        
        # Clean up any existing sphere from previous runs
        existing_sphere = bpy.data.objects.get(self.config.body_sphere_name)
        if existing_sphere:
            bpy.data.objects.remove(existing_sphere, do_unlink=True)
            print(f"    Removed existing '{self.config.body_sphere_name}'")
        
        # Get body properties - use MAX dimension for larger sphere
        body_location = original_body.matrix_world.translation.copy()
        body_parent = original_body.parent
        dims = original_body.dimensions.copy()
        max_dim = max(dims.x, dims.y, dims.z)
        
        # Create Sphere body - larger to overlap leg attachment points
        sphere_radius = max_dim * 0.75
        
        BlenderContext.deselect_all()
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=sphere_radius, 
            location=body_location,
            segments=16,
            ring_count=12
        )
        sphere_body = bpy.context.active_object
        sphere_body.name = self.config.body_sphere_name
        sphere_body[self.config.component_key] = self.config.body_tag
        
        if body_parent:
            sphere_body.parent = body_parent
            sphere_body.matrix_parent_inverse = original_body.matrix_parent_inverse.copy()
        
        # Add custom property to master controller
        master["body_shape"] = 0
        try:
            ui = master.id_properties_ui("body_shape")
            ui.update(min=0, max=1, soft_min=0, soft_max=1, 
                     description="0 = Original, 1 = Sphere", default=0)
        except:
            pass
        
        # Set initial visibility
        original_body.hide_viewport = False
        original_body.hide_render = False
        sphere_body.hide_viewport = True
        sphere_body.hide_render = True
        
        # Register visibility handler
        self._register_body_shape_handler(
            master.name, 
            original_body.name, 
            sphere_body.name
        )
        
        print(f"    Created sphere with radius {sphere_radius:.2f}")
        print(f"    Added 'body_shape' toggle to '{master.name}'")
    
    def _setup_random_walk(self):
        """
        Add random walk behavior controlled by a toggle.
        
        Adds properties to character_controller:
        - random_walk: 0 = off, 1 = on
        - walk_speed: movement speed multiplier
        - turn_speed: rotation speed multiplier
        """
        master = bpy.data.objects.get(self.config.master_controller_name)
        
        if not master:
            print("    Warning: No master controller found, skipping random walk")
            return
        
        # Add control properties
        master["random_walk"] = 0
        master["walk_speed"] = 0.05
        master["turn_speed"] = 0.02
        
        # Set up property UI
        try:
            ui = master.id_properties_ui("random_walk")
            ui.update(min=0, max=1, soft_min=0, soft_max=1,
                     description="0 = Off, 1 = On", default=0)
            
            ui = master.id_properties_ui("walk_speed")
            ui.update(min=0.0, max=0.5, soft_min=0.01, soft_max=0.2,
                     description="Movement speed", default=0.05)
            
            ui = master.id_properties_ui("turn_speed")
            ui.update(min=0.0, max=0.2, soft_min=0.005, soft_max=0.1,
                     description="Turning speed", default=0.02)
        except:
            pass
        
        # Register the frame change handler
        self._register_random_walk_handler(master.name)
        
        print(f"    Added 'random_walk' toggle to '{master.name}'")
        print(f"    Added 'walk_speed' and 'turn_speed' controls")
    
    def _register_random_walk_handler(self, master_name: str):
        """Register a frame change handler for random walk behavior."""
        handler_name = "spider_random_walk_handler"
        
        # Remove existing handler
        for handler in bpy.app.handlers.frame_change_post[:]:
            if hasattr(handler, '__name__') and handler.__name__ == handler_name:
                bpy.app.handlers.frame_change_post.remove(handler)
        
        def spider_random_walk_handler(scene):
            """Update spider position based on procedural noise."""
            master = bpy.data.objects.get(master_name)
            
            if not master:
                return
            
            # Check if random walk is enabled
            if master.get("random_walk", 0) < 0.5:
                return
            
            frame = scene.frame_current
            walk_speed = master.get("walk_speed", 0.05)
            turn_speed = master.get("turn_speed", 0.02)
            
            # Use noise for smooth random movement
            # Different seeds for X, Y, and rotation
            time_scale = 0.02  # How fast the noise changes
            
            # Noise-based direction changes
            noise_x = noise.noise(Vector((frame * time_scale, 0, 0)))
            noise_y = noise.noise(Vector((frame * time_scale, 100, 0)))
            noise_rot = noise.noise(Vector((frame * time_scale, 200, 0)))
            
            # Apply movement in the spider's forward direction
            current_rot_z = master.rotation_euler.z
            
            # Move forward with some sideways drift
            forward_x = math.cos(current_rot_z) * walk_speed
            forward_y = math.sin(current_rot_z) * walk_speed
            
            # Add noise-based drift
            drift_x = noise_x * walk_speed * 0.5
            drift_y = noise_y * walk_speed * 0.5
            
            master.location.x += forward_x + drift_x
            master.location.y += forward_y + drift_y
            
            # Rotate based on noise (creates wandering behavior)
            master.rotation_euler.z += noise_rot * turn_speed
        
        spider_random_walk_handler.__name__ = handler_name
        bpy.app.handlers.frame_change_post.append(spider_random_walk_handler)
    
    def _setup_ground_detection(self):
        """
        Set up ground detection for the spider to follow terrain.
        
        Adds properties to character_controller:
        - ground_follow: 0 = off, 1 = on
        - body_height: height of body above ground
        
        Requires a mesh object named "Ground" (or as configured) in the scene.
        The ground can be any shape - flat, bumpy, curved, etc.
        """
        master = bpy.data.objects.get(self.config.master_controller_name)
        
        if not master:
            print("    Warning: No master controller found, skipping ground detection")
            return
        
        # Add control properties
        master["ground_follow"] = 0  # Off by default (set to 1 to enable)
        master["body_height"] = 3.0  # Default height above ground
        
        # Set up property UI
        try:
            ui = master.id_properties_ui("ground_follow")
            ui.update(min=0, max=1, soft_min=0, soft_max=1,
                     description="0 = Off, 1 = On", default=0)
            
            ui = master.id_properties_ui("body_height")
            ui.update(min=0.0, max=10.0, soft_min=0.5, soft_max=5.0,
                     description="Height of body above ground", default=3.0)
        except:
            pass
        
        # Register the ground detection handler
        self._register_ground_detection_handler(
            master.name,
            self.config.ground_plane_name,
            self.config.walk_path_prefix
        )
        
        print(f"    Added 'ground_follow' toggle to '{master.name}'")
        print(f"    Ground plane name: '{self.config.ground_plane_name}'")
        print(f"    TIP: Create a mesh named '{self.config.ground_plane_name}' for terrain following")
    
    def _register_ground_detection_handler(self, master_name: str, ground_name: str, path_prefix: str):
        """Register a frame change handler for ground detection."""
        handler_name = "spider_ground_detection_handler"
        
        # Remove existing handler
        for handler in bpy.app.handlers.frame_change_post[:]:
            if hasattr(handler, '__name__') and handler.__name__ == handler_name:
                bpy.app.handlers.frame_change_post.remove(handler)
        
        def spider_ground_detection_handler(scene):
            """Raycast to ground and adjust spider height."""
            master = bpy.data.objects.get(master_name)
            ground = bpy.data.objects.get(ground_name)
            
            if not master or not ground:
                return
            
            # Check if ground follow is enabled
            if master.get("ground_follow", 0) < 0.5:
                return
            
            body_height = master.get("body_height", 3.0)
            
            # Get evaluated ground mesh for accurate raycasting
            depsgraph = bpy.context.evaluated_depsgraph_get()
            ground_eval = ground.evaluated_get(depsgraph)
            
            # Raycast from high above the master controller position
            ray_origin = master.location.copy()
            ray_origin.z = 1000  # Start very high
            ray_direction = Vector((0, 0, -1))  # Straight down
            
            # Transform ray to ground's local space
            ground_matrix_inv = ground_eval.matrix_world.inverted()
            local_origin = ground_matrix_inv @ ray_origin
            local_direction = ground_matrix_inv.to_3x3() @ ray_direction
            
            # Perform raycast
            success, location, normal, face_index = ground_eval.ray_cast(local_origin, local_direction)
            
            if success:
                # Convert hit location back to world space
                hit_world = ground_eval.matrix_world @ location
                hit_normal_world = (ground_eval.matrix_world.to_3x3() @ normal).normalized()
                
                # Set master controller height
                master.location.z = hit_world.z + body_height
                
                # Optionally tilt the spider to match ground slope
                # (commented out - can be enabled for more realism)
                # up = Vector((0, 0, 1))
                # rot_diff = up.rotation_difference(hit_normal_world)
                # master.rotation_euler = rot_diff.to_euler()
            
            # Now raycast for each walk path to adjust their heights
            for obj in bpy.data.objects:
                if obj.name.startswith(path_prefix):
                    # Get world position of the path
                    path_world_pos = obj.matrix_world.translation.copy()
                    path_world_pos.z = 1000
                    
                    # Transform to ground local space
                    local_path_origin = ground_matrix_inv @ path_world_pos
                    
                    # Raycast
                    success, location, normal, face_index = ground_eval.ray_cast(
                        local_path_origin, local_direction
                    )
                    
                    if success:
                        hit_world = ground_eval.matrix_world @ location
                        
                        # Calculate offset from master's ground level
                        master_ground_z = master.location.z - body_height
                        height_diff = hit_world.z - master_ground_z
                        
                        # Apply offset to path's local Z position
                        # This makes the feet follow local terrain variations
                        obj.location.z = height_diff
        
        spider_ground_detection_handler.__name__ = handler_name
        bpy.app.handlers.frame_change_post.append(spider_ground_detection_handler)
    
    def _add_body_noise(self):
        """Add organic movement noise to both body shapes."""
        # Apply noise to original body
        add_body_noise(self.config.body_name)
        
        # Apply noise to sphere body
        add_body_noise(self.config.body_sphere_name)
    
    def _register_body_shape_handler(self, master_name: str, body_name: str, sphere_name: str):
        """Register a depsgraph handler to update body visibility."""
        handler_name = "spider_body_shape_handler"
        for handler in bpy.app.handlers.depsgraph_update_post[:]:
            if hasattr(handler, '__name__') and handler.__name__ == handler_name:
                bpy.app.handlers.depsgraph_update_post.remove(handler)
        
        def spider_body_shape_handler(scene, depsgraph):
            master = bpy.data.objects.get(master_name)
            original = bpy.data.objects.get(body_name)
            sphere = bpy.data.objects.get(sphere_name)
            
            if not all([master, original, sphere]):
                return
            
            shape_val = master.get("body_shape", 0)
            use_sphere = shape_val >= 0.5
            
            if original.hide_viewport != use_sphere:
                original.hide_viewport = use_sphere
                original.hide_render = use_sphere
            
            if sphere.hide_viewport != (not use_sphere):
                sphere.hide_viewport = not use_sphere
                sphere.hide_render = not use_sphere
        
        spider_body_shape_handler.__name__ = handler_name
        bpy.app.handlers.depsgraph_update_post.append(spider_body_shape_handler)
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _tag_initial_components(self):
        cfg = self.config
        body = bpy.data.objects.get(cfg.body_name)
        if body:
            body[cfg.component_key] = cfg.body_tag
        for obj in bpy.data.objects:
            if obj != body and cfg.component_key not in obj:
                obj[cfg.component_key] = cfg.leg_tag
    
    def _get_tagged_objects(self, tag: str) -> List[bpy.types.Object]:
        return [obj for obj in bpy.data.objects if obj.get(self.config.component_key) == tag]
    
    def _duplicate_and_rotate(self, objects, angle_degrees, leg_index):
        BlenderContext.select_objects(objects)
        bpy.ops.object.duplicate(linked=False)
        new_objects = bpy.context.selected_objects
        
        rotation_matrix = Matrix.Rotation(math.radians(angle_degrees), 4, 'Z')
        for obj in new_objects:
            obj.matrix_world = rotation_matrix @ obj.matrix_world
            obj[self.config.component_key] = self.config.leg_tag
            obj["leg_index"] = leg_index
        
        return new_objects
    
    def _reset_path_rotations(self):
        for obj in bpy.data.objects:
            if obj.name.startswith(self.config.walk_path_prefix):
                obj.rotation_euler = (0, 0, 0)
    
    def _reassign_ik_targets_to_paths(self):
        paths_by_suffix = {}
        for obj in bpy.data.objects:
            if obj.name.startswith(self.config.walk_path_prefix):
                suffix = obj.name[len(self.config.walk_path_prefix):]
                paths_by_suffix[suffix] = obj
        
        print(f"    Found paths: {list(paths_by_suffix.keys())}")
        
        for obj in bpy.data.objects:
            if obj.name.startswith(self.config.ik_target_prefix):
                suffix = obj.name[len(self.config.ik_target_prefix):]
                matching_path = paths_by_suffix.get(suffix)
                
                if matching_path:
                    for constraint in obj.constraints:
                        if constraint.type == 'FOLLOW_PATH':
                            old_target = constraint.target.name if constraint.target else "None"
                            constraint.target = matching_path
                            print(f"    {obj.name}: {old_target} -> {matching_path.name}")
                            break
                obj.location = (0, 0, 0)
    
    def _get_leg_ik_target(self, leg_index: int):
        for obj in bpy.data.objects:
            if (obj.name.startswith(self.config.ik_target_prefix) and
                obj.get("leg_index") == leg_index):
                return obj
        
        if leg_index == 0:
            return bpy.data.objects.get(self.config.ik_target_prefix)
        else:
            suffix = f".{leg_index:03d}"
            return bpy.data.objects.get(f"{self.config.ik_target_prefix}{suffix}")
    
    def _add_constraint_if_missing(self, obj, constraint_type, target, name):
        for c in obj.constraints:
            if c.type == constraint_type and getattr(c, 'target', None) == target:
                return
        constraint = obj.constraints.new(constraint_type)
        constraint.target = target
        constraint.name = name
    
    def _shift_animation(self, action, frame_offset):
        for fcurve in action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.co.x += frame_offset
                keyframe.handle_left.x += frame_offset
                keyframe.handle_right.x += frame_offset
        
        if action.id_root == 'OBJECT':
            for obj in bpy.data.objects:
                if obj.animation_data and obj.animation_data.action == action:
                    self._shift_nla_strips(obj.animation_data, frame_offset)
    
    def _shift_nla_strips(self, anim_data, frame_offset):
        if not anim_data.nla_tracks:
            return
        for track in anim_data.nla_tracks:
            for strip in track.strips:
                strip.frame_start += frame_offset
                strip.frame_end += frame_offset


# =============================================================================
# ENTRY POINT
# =============================================================================

def build_spider(
    config: Optional[SpiderConfig] = None,
    clear_scene: bool = True
) -> bool:
    """Build a complete spider rig."""
    builder = SpiderRigBuilder(config=config)
    return builder.build(clear_scene=clear_scene)


def create_sample_ground(size: float = 30.0, subdivisions: int = 20, bumpiness: float = 1.0):
    """
    Create a sample ground plane for testing terrain following.
    
    Args:
        size: Size of the ground plane
        subdivisions: Number of subdivisions (more = smoother bumps)
        bumpiness: Height variation of the bumps (0 = flat)
    
    Returns:
        The created ground object
    """
    import bmesh
    
    # Create subdivided plane
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    
    # Subdivide
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=subdivisions)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Add random bumps using displacement
    if bumpiness > 0:
        # Use vertex displacement for bumps
        mesh = ground.data
        for vert in mesh.vertices:
            # Use noise based on XY position for smooth bumps
            noise_val = noise.noise(Vector((
                vert.co.x * 0.3,
                vert.co.y * 0.3,
                0
            )))
            vert.co.z = noise_val * bumpiness
    
    # Add a material for visibility
    mat = bpy.data.materials.new(name="Ground_Material")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.2, 0.3, 0.15, 1)
    ground.data.materials.append(mat)
    
    print(f"Created ground plane '{ground.name}' (size={size}, bumps={bumpiness})")
    return ground


def build_spider_with_ground(
    config: Optional[SpiderConfig] = None,
    ground_size: float = 30.0,
    ground_bumps: float = 1.0
) -> bool:
    """
    Build a spider rig with a sample ground plane.
    
    Args:
        config: Optional SpiderConfig
        ground_size: Size of the ground plane
        ground_bumps: Bumpiness of the ground (0 = flat)
    """
    # Build the spider
    result = build_spider(config=config, clear_scene=True)
    
    if result:
        # Create ground plane
        create_sample_ground(size=ground_size, bumpiness=ground_bumps)
        
        # Enable ground following by default
        master = bpy.data.objects.get("character_controller")
        if master:
            master["ground_follow"] = 1
            print("\nGround following enabled! Press Play to see the spider walk on terrain.")
    
    return result


if __name__ == "__main__":
    # Default build (with ground)
    build_spider_with_ground(ground_size=30.0, ground_bumps=1.5)
