import bpy
import os
import sys
import io
import traceback
import contextlib

# --- Configuration ---
WATCH_FILE = "/Users/joem/.gemini/antigravity/scratch/blender_bridge/payload.py"
CHECK_INTERVAL = 1.0

def print_to_blender_console(msg, type='INFO'):
    """Finds a Python Console in the UI and writes to it."""
    
    # 1. Try to report to the Info bar (bottom of screen) always
    # 'INFO', 'WARNING', 'ERROR'
    if type == 'Normal': type = 'INFO'
    
    # Using a helper operator context is tricky for reports from a timer, 
    # but print() usually goes to system console.
    # We will try to find the CONSOLE area.
    
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'CONSOLE':
                # Context override to print to this console
                with bpy.context.temp_override(window=window, area=area):
                    # Write separate lines
                    for line in str(msg).split('\n'):
                        bpy.ops.console.scrollback_append(text=line, type=type)
                return

class AntigravityBridgeOperator(bpy.types.Operator):
    """Antigravity Bridge: Watches a file and executes it."""
    bl_idname = "wm.antigravity_bridge"
    bl_label = "Start Antigravity Bridge"
    
    _timer = None
    _last_mtime = 0
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            if os.path.exists(WATCH_FILE):
                try:
                    mtime = os.stat(WATCH_FILE).st_mtime
                    if mtime > self._last_mtime:
                        self._last_mtime = mtime
                        self.execute_external_script(context)
                except OSError:
                    pass
        return {'PASS_THROUGH'}

    def execute_external_script(self, context):
        print_to_blender_console(f"--- DETECTED CHANGE: {os.path.basename(WATCH_FILE)} ---", 'OUTPUT')
        
        try:
            with open(WATCH_FILE, 'r') as f:
                code = f.read()
            
            # Setup namespace
            namespace = globals().copy()
            namespace['context'] = context
            
            # Capture stdout/stderr
            f_out = io.StringIO()
            
            with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_out):
                exec(code, namespace)
            
            # Print output
            output = f_out.getvalue()
            if output:
                print_to_blender_console(output, 'OUTPUT')
            
            print_to_blender_console("Execution Successful.", 'INFO')
            
            # Redraw
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()

        except Exception:
            # Capture traceback
            tb = traceback.format_exc()
            print_to_blender_console(tb, 'ERROR')
            print_to_blender_console("Execution Failed.", 'ERROR')

    def execute(self, context):
        os.makedirs(os.path.dirname(WATCH_FILE), exist_ok=True)
        if not os.path.exists(WATCH_FILE):
            with open(WATCH_FILE, 'w') as f:
                f.write(f"print('Bridge Connected! Blender Version: {bpy.app.version_string}')")

        try:
            self._last_mtime = os.stat(WATCH_FILE).st_mtime
        except OSError:
            self._last_mtime = 0

        wm = context.window_manager
        self._timer = wm.event_timer_add(CHECK_INTERVAL, window=context.window)
        wm.modal_handler_add(self)
        
        msg = f"Antigravity Bridge Started. Connected to {WATCH_FILE}"
        self.report({'INFO'}, msg)
        print_to_blender_console(msg, 'INFO')
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        print_to_blender_console("Antigravity Bridge Stopped.", 'INFO')

def register():
    bpy.utils.register_class(AntigravityBridgeOperator)

def unregister():
    bpy.utils.unregister_class(AntigravityBridgeOperator)

if __name__ == "__main__":
    register()
    # Attempt stop old one by replacing? No easy way, just start new.
    # User might need to restart blender if multiple timers stack, but usually fine.
    try:
        bpy.ops.wm.antigravity_bridge()
    except Exception:
        pass
