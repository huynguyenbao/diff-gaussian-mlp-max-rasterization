

// __device__ void backward_mlp(
//     const float3 &g_xyz_vs,
//     const float3 &pixDir,
//     const float3x3 &g_inv_mat_modelview,
//     const float8x3 &weight1,
//     const float8 &bias1,
//     const float8 &weight2,
//     const float bias2,
//     const float &integral,
//     const float &dL_dintegral,
//     float3 &dL_dg_xyz_vs,
//     float3x3 &dL_dg_inv_mat_modelview,
//     float8x3 &dL_dweight1,
//     float8 &dL_dbias1,
//     float8 &dL_dweight2,
//     float &dL_dbias2)
// {
//     // Initialize gradients to zero
//     dL_dg_xyz_vs = make_float3(0, 0, 0);
//     dL_dg_inv_mat_modelview = make_float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
//     dL_dweight1 = make_float8x3(0);
//     dL_dbias1 = make_float8(0);
//     dL_dweight2 = make_float8(0);
//     dL_dbias2 = 0.0f;

//     // Skip if forward had no contribution
//     if (integral <= 0.0f)
//         return;

//     // Recompute forward intermediates
//     float3 dir_world = g_inv_mat_modelview * pixDir;
//     float3 g_xyz_ws = g_inv_mat_modelview * g_xyz_vs;
//     float A_val = dot(dir_world, dir_world);
//     float A = max(EPS, A_val);
//     float B = dot(g_xyz_ws, dir_world);
//     float C_val = dot(g_xyz_ws, g_xyz_ws) - 1;
//     float D = B * B - A * C_val;
//     if (D < 0.0f)
//         return;

//     float sqrtD = sqrtf(D);
//     float t0 = (B - sqrtD) / A;
//     float t1 = (B + sqrtD) / A;

//     float3 d_tilde = dir_world; // Same as dir_world
//     float3 o_tilde = -g_xyz_ws; // -1 * g_xyz_ws

//     float length_d = sqrtf(dot(d_tilde, d_tilde));
//     float8 w1_tilde;
//     float8 b1_tilde;
//     for (int i = 0; i < 8; i++)
//     {
//         w1_tilde.data[i] = dot(weight1.rows[i], d_tilde);
//         b1_tilde.data[i] = dot(weight1.rows[i], o_tilde) + bias1.data[i];
//     }

//     // Compute unscaled integral (integral_sum)
//     float integral_sum = bias2 * (t1 - t0);
//     for (int i = 0; i < 8; i++)
//     {
//         float denom = w1_tilde.data[i] + EPS;
//         float phi1 = b1_tilde.data[i] + w1_tilde.data[i] * t1;
//         float phi0 = b1_tilde.data[i] + w1_tilde.data[i] * t0;
//         integral_sum += weight2.data[i] / denom * (sinf(phi1) - sinf(phi0));
//     }

//     // Backpropagate scaling by length_d
//     float dL_dintegral_sum = dL_dintegral * length_d;
//     float dL_dlength_d = dL_dintegral * integral_sum;

//     // Backpropagate through integral_sum
//     dL_dbias2 = dL_dintegral_sum * (t1 - t0);
//     float dL_dt1 = dL_dintegral_sum * bias2;
//     float dL_dt0 = -dL_dintegral_sum * bias2;

//     float8 dL_dw1_tilde = make_float8(0);
//     float8 dL_db1_tilde = make_float8(0);
//     float8 dL_dweight2_local = make_float8(0);

//     for (int i = 0; i < 8; i++)
//     {
//         float denom = w1_tilde.data[i] + EPS;
//         float w2_i = weight2.data[i];
//         float phi1 = b1_tilde.data[i] + w1_tilde.data[i] * t1;
//         float phi0 = b1_tilde.data[i] + w1_tilde.data[i] * t0;
//         float sin_phi1 = sinf(phi1);
//         float sin_phi0 = sinf(phi0);
//         float diff_sin = sin_phi1 - sin_phi0;
//         float cos_phi1 = cosf(phi1);
//         float cos_phi0 = cosf(phi0);

//         // Gradient for weight2
//         dL_dweight2_local.data[i] = dL_dintegral_sum * diff_sin / denom;

//         // Gradient for denominator
//         float dL_ddenom = dL_dintegral_sum * w2_i * (-diff_sin) / (denom * denom);

//         // Gradient for w1_tilde (from denominator)
//         dL_dw1_tilde.data[i] = dL_ddenom;

//         // Gradients from trigonometric terms
//         float dL_dphi1 = dL_dintegral_sum * w2_i / denom * cos_phi1;
//         float dL_dphi0 = dL_dintegral_sum * w2_i / denom * (-cos_phi0);

//         // Gradient for b1_tilde and w1_tilde (from phis)
//         dL_db1_tilde.data[i] += dL_dphi1 + dL_dphi0;
//         dL_dw1_tilde.data[i] += dL_dphi1 * t1 + dL_dphi0 * t0;

//         // Gradient for t0 and t1
//         dL_dt1 += dL_dphi1 * w1_tilde.data[i];
//         dL_dt0 += dL_dphi0 * w1_tilde.data[i];
//     }

//     // Backpropagate length_d to d_tilde
//     float3 dL_dd_tilde = (length_d > 0.0f) ? (dL_dlength_d * d_tilde / length_d) : make_float3(0, 0, 0);

//     // Backpropagate MLP intermediates
//     float3 dL_do_tilde = make_float3(0, 0, 0);
//     for (int i = 0; i < 8; i++)
//     {
//         // Gradient for weight1
//         dL_dweight1.rows[i] =
//             dL_dw1_tilde.data[i] * d_tilde +
//             dL_db1_tilde.data[i] * o_tilde;

//         // Gradient for bias1
//         dL_dbias1.data[i] = dL_db1_tilde.data[i];

//         // Accumulate gradients for d_tilde and o_tilde
//         dL_dd_tilde += dL_dw1_tilde.data[i] * weight1.rows[i];
//         dL_do_tilde += dL_db1_tilde.data[i] * weight1.rows[i];
//     }

//     // Backpropagate to transformation matrix
//     dL_dg_inv_mat_modelview =
//         outer_product(dL_dd_tilde, pixDir) -
//         outer_product(dL_do_tilde, g_xyz_vs);

//     // Backpropagate to Gaussian position
//     dL_dg_xyz_vs = -transpose(g_inv_mat_modelview) * dL_do_tilde;

//     // Backpropagate t0, t1 to quadratic parameters
//     float invA = 1.0f / A;
//     float dL_dA = dL_dt0 * (-t0 * invA) + dL_dt1 * (-t1 * invA);
//     float dL_dB = (dL_dt0 + dL_dt1) * invA;
//     float dL_dD = (dL_dt1 - dL_dt0) * (0.5f / (A * sqrtD));

//     dL_dB += 2.0f * B * dL_dD;
//     dL_dA += -C_val * dL_dD;
//     float dL_dC_val = -A * dL_dD;

//     // Backpropagate A, B, C to intermediates
//     float dL_dA_val = (A_val > EPS) ? dL_dA : 0.0f;
//     float3 dL_ddir_world = 2.0f * dL_dA_val * dir_world + dL_dB * g_xyz_ws;
//     float3 dL_dg_xyz_ws = dL_dB * dir_world + 2.0f * dL_dC_val * g_xyz_ws;

//     // Combine gradients for transformation matrix
//     dL_dg_inv_mat_modelview +=
//         outer_product(dL_ddir_world, pixDir) +
//         outer_product(dL_dg_xyz_ws, g_xyz_vs);

//     // Combine gradients for Gaussian position
//     dL_dg_xyz_vs += transpose(g_inv_mat_modelview) * dL_dg_xyz_ws;

//     // Combine weight2 gradients
//     dL_dweight2 = dL_dweight2_local;
// }