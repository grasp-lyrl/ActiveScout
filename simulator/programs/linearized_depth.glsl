#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;

out vec2 uv;

void main() {
    gl_Position = vec4(in_position, 1.0);
    uv = in_texcoord_0;
}

#elif defined FRAGMENT_SHADER

layout(location=0) out vec4 out_color;
in vec2 uv;

uniform sampler2D texture0;
uniform float near;
uniform float far;

float LinearizeDepth(in vec2 uv)
{
    float depth = texture(texture0, uv).r;
    return (2.0 * near) / (far + near - depth * (far - near));
}

void main() {
    float d = LinearizeDepth(uv);
    out_color = vec4(d);
}
#endif
