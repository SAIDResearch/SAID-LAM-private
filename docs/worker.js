export default {
    async fetch(request, env) {
        const url = new URL(request.url);
        const method = request.method;

        // Helper: Generate a BETA license key
        const generateKey = () => "BETA_" + crypto.randomUUID().replace(/-/g, "").substring(0, 16).toUpperCase();

        // 1. /register - Device-locked registration
        if (url.pathname === "/register" && method === "POST") {
            const { email, mac_address } = await request.json();
            if (!email || !mac_address) return Response.json({ error: "Missing fields" }, { status: 400 });

            // Check if this MAC already has a key
            const existingKey = await env.KV.get(`mac:${mac_address}`);
            if (existingKey) {
                const data = await env.KV.get(`beta:${existingKey}`, { type: "json" });

                // Renewal Logic: If expired or revoked, we could allow the user to "reset" 
                // For now, let's just allow them to extend the expiry if they register again
                const now = new Date();
                const expiry = new Date(data.expires_at);

                if (now > expiry || data.revoked) {
                    const newExpiresAt = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
                    data.expires_at = newExpiresAt;
                    data.revoked = false;
                    await env.KV.put(`beta:${existingKey}`, JSON.stringify(data));
                    return Response.json({
                        license_key: existingKey,
                        expires_at: newExpiresAt,
                        tier: "BETA",
                        status: "renewed"
                    });
                }

                return Response.json({
                    license_key: existingKey,
                    expires_at: data.expires_at,
                    tier: "BETA",
                    status: "already_active"
                });
            }

            // Create new 30-day license
            const key = generateKey();
            const expiresAt = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];

            const licenseData = {
                email,
                mac_address,
                expires_at: expiresAt,
                created_at: new Date().toISOString(),
                revoked: false
            };

            await env.KV.put(`beta:${key}`, JSON.stringify(licenseData));
            await env.KV.put(`mac:${mac_address}`, key); // Device lock

            return Response.json({
                license_key: key,
                expires_at: expiresAt,
                tier: "BETA"
            });
        }

        // 2. /validate - Key + MAC validation
        if (url.pathname === "/validate" && method === "POST") {
            const { license_key, mac_address } = await request.json();
            const data = await env.KV.get(`beta:${license_key}`, { type: "json" });

            if (!data) return Response.json({ valid: false });

            // Revocation Check
            if (data.revoked) return Response.json({ valid: false, reason: "revoked" });

            // MAC Check
            if (data.mac_address !== mac_address) return Response.json({ valid: false, reason: "mac_mismatch" });

            // Expiry Check
            if (new Date() > new Date(data.expires_at)) return Response.json({ valid: false });

            return Response.json({ valid: true });
        }

        // 3. /request-beta - Simple status check
        if (url.pathname === "/request-beta" && method === "POST") {
            return Response.json({ status: "approved" }); // Auto-approve for POC
        }

        return new Response("SAID-LAM License Server", { status: 200 });
    }
};
