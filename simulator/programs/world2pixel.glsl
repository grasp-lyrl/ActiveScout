#version 330

#if defined VERTEX_SHADER

in vec3 in_position;

uniform vec3 position;
uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(in_position+position, 1.0);
}

#elif defined FRAGMENT_SHADER

out vec4 outColor;
uniform vec4 color;

void main() {
    outColor = color;
}

#endif
