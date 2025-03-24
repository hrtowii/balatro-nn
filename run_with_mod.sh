#!/bin/bash

# LOVE_EXECUTABLE="$HOME/Library/Application Support/Steam/steamapps/common/Balatro/Balatro.app/Contents/MacOS/love"
# LOVE_ORIGINAL="$HOME/Library/Application Support/Steam/steamapps/common/Balatro/Balatro.app/Contents/MacOS/love_original"
# LOVE_PATCHED="$HOME/Library/Application Support/Steam/steamapps/common/Balatro/Balatro.app/Contents/MacOS/love_patched"
# DYLIB_PATH="/Users/ibarahime/Downloads/lovely/liblovely.dylib"
# INSERT_DYLIB="/Users/ibarahime/dev/insert_dylib/insert_dylib/insert_dylib"

# if [ ! -f "$LOVE_ORIGINAL" ]; then
#     echo "Creating patched version of Balatro..."

#     cp "$LOVE_EXECUTABLE" "$LOVE_ORIGINAL"

#     "$INSERT_DYLIB" "$DYLIB_PATH" "$LOVE_EXECUTABLE" --all-yes

#     mv "$LOVE_PATCHED" "$LOVE_EXECUTABLE"

#     echo "Patching complete! Balatro is now patched."
# else
#     echo "Patched version already exists."
# fi

# Define directories
MODS_DIR="$HOME/Library/Application Support/Balatro/Mods"
MOD_DIR="$MODS_DIR/balatro_nn"
BALATROBOT_DIR="$MODS_DIR/balatrobot"
SOURCE_DIR="./game_interaction"
SOURCE_BALATROBOT_DIR="./balatrobot"

# Create and update balatro_nn mod directory
if [ ! -d "$MOD_DIR" ]; then
    echo "Creating mod directory at $MOD_DIR..."
    mkdir -p "$MOD_DIR"
else
    echo "Mod directory already exists. Removing old files..."
    rm -rf "$MOD_DIR"/*
fi

echo "Copying balatro_nn files to mod directory..."
cp -R "$SOURCE_DIR"/src "$MOD_DIR/"
cp "$SOURCE_DIR"/main.lua "$MOD_DIR/"
# Create and update balatrobot directory in parent Mods folder
if [ ! -d "$BALATROBOT_DIR" ]; then
    echo "Creating balatrobot directory at $BALATROBOT_DIR..."
    mkdir -p "$BALATROBOT_DIR"
else
    echo "Balatrobot directory already exists. Removing old files..."
    rm -rf "$BALATROBOT_DIR"/*
fi

echo "Copying balatrobot files to parent Mods directory..."
cp -R "$SOURCE_BALATROBOT_DIR"/* "$BALATROBOT_DIR/"
echo "Mod files updated successfully."

# Launch the patched executable
# echo "Launching patched Balatro..."
pwd=pwd
game="$HOME/Library/Application Support/Steam/steamapps/common/Balatro"
export DYLD_INSERT_LIBRARIES=liblovely.dylib
cd "$game"
./Balatro.app/Contents/MacOS/love "$@"