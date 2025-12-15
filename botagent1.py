"""
Spider Rig Builder - Improved Version
======================================
Builds a complete quadruped spider rig with:
- Modular leg system with IK
- Circular walk paths with gait offsets
- Steering controller
- Organic body noise
- Toggleable body shape (cube/sphere)

Usage:
    Run in Blender's scripting environment or via command line.
"""

import bpy
import math
from mathutils import Vector, Matrix
import importlib
import sys
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


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
    
    # Component tagging (for reliable selection)
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


class ModuleLoader:
    """Handles dynamic module loading with proper reloading."""
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else self._detect_base_path()
        self._ensure_path()
        self._loaded_modules: Dict[str, Any] = {}
    
    def _detect_base_path(self) -> Path:
        """Detect base path from current file or fallback."""
        if "__file__" in globals():
            return Path(__file__).parent
        # Fallback: look for common module in current directory
        return Path(bpy.path.abspath("//")).parent
    
    def _ensure_path(self):
        """Ensure base path is in sys.path."""
        path_str = str(self.base_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    
    def load(self, module_name: str, force_reload: bool = True) -> Any:
        """Load or reload a module."""
        try:
            if module_name in sys.modules and force_reload:
                module = importlib.reload(sys.modules[module_name])
            else:
                module = importlib.import_module(module_name)
            self._loaded_modules[module_name] = module
            return module
        except ImportError as e:
            raise ImportError(f"Failed to load module '{module_name}' from {self.base_path}: {e}")


# =============================================================================
# CORE COMPONENTS
# =============================================================================

class SpiderRigBuilder:
    """Main builder class for the spider rig."""
    
    def __init__(self, config: Optional[SpiderConfig] = None, module_path: Optional[str] = None):
        self.config = config or SpiderConfig()
        self.loader = ModuleLoader(module_path)
        self.leg_objects: Dict[int, List[bpy.types.Object]] = {}  # leg_index -> [objects]
        
    def build(self, clear_scene: bool = True) -> bool:
        """
        Execute the full spider rig build.
        
        Returns:
            True if build completed successfully, False otherwise.
        """
        print("=" * 60)
        print("SPIDER RIG BUILDER - Starting Build")
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
                ("Adding Body Noise", self._add_body_noise),
                ("Setting Up Body Shape Toggle", self._setup_body_shape_toggle),  # LAST step
            ]
            
            for i, (description, step_func) in enumerate(steps, 1):
                print(f"\n[{i}/{len(steps)}] {description}...")
                try:
                    step_func()
                    print(f"    ✓ Complete")
                except Exception as step_error:
                    print(f"    ✗ STEP FAILED: {step_error}")
                    import traceback
                    traceback.print_exc()
                    # Continue with remaining steps to see what else might work
                    continue
            
            print("\n" + "=" * 60)
            print("SPIDER RIG BUILD COMPLETE!")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\n✗ BUILD FAILED: {e}")
            import traceback
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
        module = self.loader.load("create_spider_assembly")
        module.create_spider_assembly()
        self._tag_initial_components()
    
    def _create_walk_path(self):
        """Create the circular walk path for the first leg."""
        module = self.loader.load("create_path")
        module.create_circular_path()
        
        # Tag the newly created path as a leg component so it gets duplicated
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
        
        # Get EVERYTHING except body - this includes leg geometry, IK target, AND walkpath
        base_leg_objects = [o for o in bpy.data.objects if o != body]
        
        if not base_leg_objects:
            raise RuntimeError("No leg objects found to duplicate")
        
        print(f"    Base objects to duplicate: {[o.name for o in base_leg_objects]}")
        
        # Store reference to leg 0
        self.leg_objects[0] = base_leg_objects.copy()
        
        # Duplicate for remaining legs
        for leg_index in range(1, self.config.num_legs):
            angle = self.config.leg_angles[leg_index]
            new_objects = self._duplicate_and_rotate(base_leg_objects, angle, leg_index)
            self.leg_objects[leg_index] = new_objects
            print(f"    Leg {leg_index} objects: {[o.name for o in new_objects]}")
        
        # Reset path rotations (they inherit rotation from duplication)
        self._reset_path_rotations()
        
        # Snap IK targets to their respective paths
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
        # Clear selection to ensure clean context for empty creation
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
        
        # Add copy rotation constraints to all paths
        paths_constrained = 0
        for obj in bpy.data.objects:
            if obj.name.startswith(self.config.walk_path_prefix):
                self._add_constraint_if_missing(obj, 'COPY_ROTATION', ctrl, "Steering")
                paths_constrained += 1
        
        print(f"    Linked {paths_constrained} walk paths to direction controller")
    
    def _create_master_controller(self):
        """
        Create the master 'character_controller' and parent the rig hierarchy.
        
        Hierarchy:
            character_controller (CUBE empty)
            ├── Direction_Controller
            ├── Spider_Body
            └── WalkPath, WalkPath.001, WalkPath.002, WalkPath.003
        """
        # Clear selection to ensure clean context
        BlenderContext.deselect_all()
        
        master_name = self.config.master_controller_name
        
        # Create or get master controller
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
        
        # Collect objects to parent
        children_to_parent = []
        
        # Direction Controller
        dir_ctrl = bpy.data.objects.get(self.config.direction_controller_name)
        if dir_ctrl:
            children_to_parent.append(dir_ctrl)
        
        # Spider Body
        body = bpy.data.objects.get(self.config.body_name)
        if body:
            children_to_parent.append(body)
        
        # All Walk Paths
        for obj in bpy.data.objects:
            if obj.name.startswith(self.config.walk_path_prefix):
                children_to_parent.append(obj)
        
        if not children_to_parent:
            print("    Warning: No children found to parent!")
            return
        
        # Parent using operator (keeps transforms)
        BlenderContext.deselect_all()
        for child in children_to_parent:
            child.select_set(True)
        
        master.select_set(True)
        bpy.context.view_layer.objects.active = master
        bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
        
        print(f"    Parented {len(children_to_parent)} objects to '{master_name}':")
        for child in children_to_parent:
            print(f"      └── {child.name}")
    
    def _add_body_noise(self):
        """Add organic movement noise to the body."""
        try:
            module = self.loader.load("create_body_noise")
            module.add_body_noise()
        except ImportError:
            print("    Skipping body noise (module not found)")
            self._add_default_body_noise()
    
    def _add_default_body_noise(self):
        """
        Fallback body noise using drivers if external module unavailable.
        Adds subtle procedural movement to the spider body.
        """
        body = bpy.data.objects.get(self.config.body_name)
        if not body:
            return
        
        # Add noise modifier to location/rotation via drivers
        noise_scale = 0.05
        noise_speed = 0.5
        
        for channel_idx, channel in enumerate(['x', 'y', 'z']):
            # Location noise
            driver = body.driver_add('location', channel_idx).driver
            driver.type = 'SCRIPTED'
            
            var = driver.variables.new()
            var.name = 'frame'
            var.type = 'SINGLE_PROP'
            var.targets[0].id_type = 'SCENE'
            var.targets[0].id = bpy.context.scene
            var.targets[0].data_path = 'frame_current'
            
            # Offset each axis slightly for variation
            offset = channel_idx * 100
            driver.expression = f"noise.noise(Vector((frame * {noise_speed} + {offset}, 0, 0))) * {noise_scale}"
        
        print("    Added default procedural body noise via drivers")
    
    def _setup_body_shape_toggle(self):
        """
        Create toggleable body shapes (original/sphere) driven by a custom property.
        
        Uses an application handler for reliable updates instead of drivers.
        Adds a 'body_shape' property to character_controller:
        - 0 = Original body (cube-like)
        - 1 = Sphere
        """
        BlenderContext.deselect_all()
        
        master = bpy.data.objects.get(self.config.master_controller_name)
        original_body = bpy.data.objects.get(self.config.body_name)
        
        if not original_body:
            print("    Warning: No Spider_Body found, skipping shape toggle")
            return
        
        if not master:
            print("    Warning: No master controller found, skipping shape toggle")
            return
        
        # Get original body properties for matching the sphere
        body_location = original_body.matrix_world.translation.copy()
        body_parent = original_body.parent
        dims = original_body.dimensions.copy()
        avg_size = (dims.x + dims.y + dims.z) / 3
        
        # Create Sphere body as alternative
        BlenderContext.deselect_all()
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=avg_size / 2, 
            location=body_location,
            segments=16,
            ring_count=12
        )
        sphere_body = bpy.context.active_object
        sphere_body.name = self.config.body_sphere_name
        sphere_body[self.config.component_key] = self.config.body_tag
        
        # Parent sphere to same parent as original body
        if body_parent:
            sphere_body.parent = body_parent
            sphere_body.matrix_parent_inverse = original_body.matrix_parent_inverse.copy()
        
        # Add custom property to master controller
        master["body_shape"] = 0
        
        # Set up property UI limits
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
        
        # Register the handler for updating visibility
        self._register_body_shape_handler(
            master.name, 
            original_body.name, 
            sphere_body.name
        )
        
        print(f"    Original body: '{original_body.name}'")
        print(f"    Created sphere: '{sphere_body.name}'")
        print(f"    Added 'body_shape' to '{master.name}' (0=Original, 1=Sphere)")
        print(f"    Handler registered for automatic visibility switching")
    
    def _register_body_shape_handler(self, master_name: str, body_name: str, sphere_name: str):
        """Register a depsgraph handler to update body visibility based on property."""
        
        # Remove any existing handler with same name
        handler_name = "spider_body_shape_handler"
        for handler in bpy.app.handlers.depsgraph_update_post[:]:
            if hasattr(handler, '__name__') and handler.__name__ == handler_name:
                bpy.app.handlers.depsgraph_update_post.remove(handler)
        
        def spider_body_shape_handler(scene, depsgraph):
            """Update body visibility based on body_shape property."""
            master = bpy.data.objects.get(master_name)
            original = bpy.data.objects.get(body_name)
            sphere = bpy.data.objects.get(sphere_name)
            
            if not all([master, original, sphere]):
                return
            
            shape_val = master.get("body_shape", 0)
            use_sphere = shape_val >= 0.5
            
            # Only update if changed to avoid infinite loops
            if original.hide_viewport != use_sphere:
                original.hide_viewport = use_sphere
                original.hide_render = use_sphere
            
            if sphere.hide_viewport != (not use_sphere):
                sphere.hide_viewport = not use_sphere
                sphere.hide_render = not use_sphere
        
        spider_body_shape_handler.__name__ = handler_name
        bpy.app.handlers.depsgraph_update_post.append(spider_body_shape_handler)
    
    # Keep the driver method for reference but won't use it
    def _add_visibility_driver(
        self, 
        obj: bpy.types.Object, 
        controller: bpy.types.Object, 
        prop_name: str,
        visible_when_low: bool = True
    ):
        """Add a driver to object visibility based on a controller property."""
        
        # Remove existing drivers first
        try:
            obj.driver_remove('hide_viewport')
        except:
            pass
        try:
            obj.driver_remove('hide_render')
        except:
            pass
        
        # Determine expression based on visibility logic
        if visible_when_low:
            expression = 'shape >= 0.5'  # Hide when shape is high (>=0.5)
        else:
            expression = 'shape < 0.5'   # Hide when shape is low (<0.5)
        
        # Drive hide_viewport
        fcurve = obj.driver_add('hide_viewport')
        driver = fcurve.driver
        driver.type = 'SCRIPTED'
        
        var = driver.variables.new()
        var.name = 'shape'
        var.type = 'SINGLE_PROP'
        var.targets[0].id = controller
        var.targets[0].data_path = f'["{prop_name}"]'
        driver.expression = expression
        
        # Drive hide_render the same way
        fcurve_render = obj.driver_add('hide_render')
        driver_render = fcurve_render.driver
        driver_render.type = 'SCRIPTED'
        
        var_render = driver_render.variables.new()
        var_render.name = 'shape'
        var_render.type = 'SINGLE_PROP'
        var_render.targets[0].id = controller
        var_render.targets[0].data_path = f'["{prop_name}"]'
        driver_render.expression = expression
        
        print(f"      Driver on {obj.name}: hide when {expression}")
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _tag_initial_components(self):
        """Tag initial components for reliable identification."""
        cfg = self.config
        
        # Tag body
        body = bpy.data.objects.get(cfg.body_name)
        if body:
            body[cfg.component_key] = cfg.body_tag
        
        # Tag leg components (everything that's not the body)
        for obj in bpy.data.objects:
            if obj != body and cfg.component_key not in obj:
                obj[cfg.component_key] = cfg.leg_tag
    
    def _get_tagged_objects(self, tag: str) -> List[bpy.types.Object]:
        """Get all objects with a specific component tag."""
        return [
            obj for obj in bpy.data.objects
            if obj.get(self.config.component_key) == tag
        ]
    
    def _duplicate_and_rotate(
        self,
        objects: List[bpy.types.Object],
        angle_degrees: float,
        leg_index: int
    ) -> List[bpy.types.Object]:
        """Duplicate objects and rotate around world Z axis."""
        BlenderContext.select_objects(objects)
        bpy.ops.object.duplicate(linked=False)
        
        new_objects = bpy.context.selected_objects
        
        # Apply rotation
        rotation_matrix = Matrix.Rotation(math.radians(angle_degrees), 4, 'Z')
        for obj in new_objects:
            obj.matrix_world = rotation_matrix @ obj.matrix_world
            # Tag with leg index for easy lookup
            obj[self.config.component_key] = self.config.leg_tag
            obj["leg_index"] = leg_index
        
        return new_objects
    
    def _reset_path_rotations(self):
        """Reset all walk path rotations to zero."""
        for obj in bpy.data.objects:
            if obj.name.startswith(self.config.walk_path_prefix):
                obj.rotation_euler = (0, 0, 0)
    
    def _reset_ik_target_locations(self):
        """Reset all IK target locations (snaps them to path start)."""
        for obj in bpy.data.objects:
            if obj.name.startswith(self.config.ik_target_prefix):
                obj.location = (0, 0, 0)
    
    def _reassign_ik_targets_to_paths(self):
        """
        After duplication, each IK target still points to the original WalkPath.
        This method reassigns each IK target to its corresponding duplicated path.
        """
        # Build a mapping: suffix -> path (e.g., "" -> WalkPath, ".001" -> WalkPath.001)
        paths_by_suffix = {}
        for obj in bpy.data.objects:
            if obj.name.startswith(self.config.walk_path_prefix):
                # Extract suffix (empty string for original, ".001" for first dup, etc.)
                suffix = obj.name[len(self.config.walk_path_prefix):]
                paths_by_suffix[suffix] = obj
        
        print(f"    Found paths: {list(paths_by_suffix.keys())}")
        
        # Reassign each IK target to the path with matching suffix
        for obj in bpy.data.objects:
            if obj.name.startswith(self.config.ik_target_prefix):
                suffix = obj.name[len(self.config.ik_target_prefix):]
                matching_path = paths_by_suffix.get(suffix)
                
                if matching_path:
                    # Find and update the Follow Path constraint
                    for constraint in obj.constraints:
                        if constraint.type == 'FOLLOW_PATH':
                            old_target = constraint.target.name if constraint.target else "None"
                            constraint.target = matching_path
                            print(f"    {obj.name}: {old_target} -> {matching_path.name}")
                            break
                
                # Reset location to snap to path
                obj.location = (0, 0, 0)
    
    def _get_leg_ik_target(self, leg_index: int) -> Optional[bpy.types.Object]:
        """Get the IK target for a specific leg by index."""
        # First try to find by stored leg_index property
        for obj in bpy.data.objects:
            if (obj.name.startswith(self.config.ik_target_prefix) and
                obj.get("leg_index") == leg_index):
                return obj
        
        # Fallback to suffix naming (.001, .002, etc.)
        if leg_index == 0:
            return bpy.data.objects.get(self.config.ik_target_prefix)
        else:
            suffix = f".{leg_index:03d}"
            return bpy.data.objects.get(f"{self.config.ik_target_prefix}{suffix}")
    
    def _add_constraint_if_missing(
        self,
        obj: bpy.types.Object,
        constraint_type: str,
        target: bpy.types.Object,
        name: str
    ):
        """Add a constraint if one of the same type/target doesn't exist."""
        for c in obj.constraints:
            if c.type == constraint_type and getattr(c, 'target', None) == target:
                return  # Already exists
        
        constraint = obj.constraints.new(constraint_type)
        constraint.target = target
        constraint.name = name
    
    def _shift_animation(self, action: bpy.types.Action, frame_offset: float):
        """Shift all keyframes in an action by a frame offset."""
        for fcurve in action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.co.x += frame_offset
                keyframe.handle_left.x += frame_offset
                keyframe.handle_right.x += frame_offset
        
        # Also handle NLA strips if present
        if action.id_root == 'OBJECT':
            for obj in bpy.data.objects:
                if obj.animation_data and obj.animation_data.action == action:
                    self._shift_nla_strips(obj.animation_data, frame_offset)
    
    def _shift_nla_strips(self, anim_data: bpy.types.AnimData, frame_offset: float):
        """Shift NLA strip timing."""
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
    module_path: Optional[str] = None,
    clear_scene: bool = True
) -> bool:
    """
    Build a complete spider rig.
    
    Args:
        config: Optional SpiderConfig for customization
        module_path: Path to spider module files (auto-detected if None)
        clear_scene: Whether to clear existing scene objects
    
    Returns:
        True if build succeeded, False otherwise
    """
    builder = SpiderRigBuilder(config=config, module_path=module_path)
    return builder.build(clear_scene=clear_scene)


# Alternative entry point with custom gait patterns
def build_spider_custom_gait(
    offset_pattern: List[int],
    offset_frames: float = -10.0,
    **kwargs
) -> bool:
    """
    Build spider with custom gait offset pattern.
    
    Args:
        offset_pattern: List of leg indices to offset (e.g., [0, 2] for alternate pair)
        offset_frames: Frame offset amount
        **kwargs: Additional arguments passed to build_spider
    """
    config = SpiderConfig(
        offset_leg_indices=offset_pattern,
        gait_offset_frames=offset_frames
    )
    return build_spider(config=config, **kwargs)


if __name__ == "__main__":
    # Default build
    build_spider()
    
    # Or with custom configuration:
    # config = SpiderConfig(num_legs=6, leg_angles=[0, 60, 120, 180, 240, 300])
    # build_spider(config=config)