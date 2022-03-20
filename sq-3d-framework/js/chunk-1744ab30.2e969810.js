(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-1744ab30"],{"1b53":function(e,t,r){"use strict";r.d(t,"b",(function(){return n})),r.d(t,"a",(function(){return a}));var i=r("5a89");class n{constructor(){this.enabled=!0,this.needsSwap=!0,this.clear=!1,this.renderToScreen=!1}setSize(){}render(){console.error("THREE.Pass: .render() must be implemented in derived pass.")}}const o=new i["D"](-1,1,1,-1,0,1),s=new i["c"];s.setAttribute("position",new i["p"]([-1,3,0,-1,-1,0,3,-1,0],3)),s.setAttribute("uv",new i["p"]([0,2,0,0,2,0],2));class a{constructor(e){this._mesh=new i["y"](s,e)}dispose(){this._mesh.geometry.dispose()}render(e){e.render(this._mesh,o)}get material(){return this._mesh.material}set material(e){this._mesh.material=e}}},"232b":function(e,t,r){"use strict";r.d(t,"b",(function(){return i})),r.d(t,"a",(function(){return h}));var i,n=r("3835"),o=r("d4ec"),s=r("bee2"),a=r("ade3"),c=(r("d3b7"),r("159b"),r("5a89")),l=r("4e0a");(function(e){e[e["RECT"]=0]="RECT",e[e["CIRCLE"]=1]="CIRCLE"})(i||(i={}));var h=function(){function e(){var t=arguments.length>0&&void 0!==arguments[0]?arguments[0]:100,r=arguments.length>1&&void 0!==arguments[1]?arguments[1]:100,n=arguments.length>2?arguments[2]:void 0,s=arguments.length>3?arguments[3]:void 0,l=arguments.length>4?arguments[4]:void 0,h=arguments.length>5?arguments[5]:void 0,d=arguments.length>6?arguments[6]:void 0,v=arguments.length>7&&void 0!==arguments[7]?arguments[7]:i.RECT;Object(o["a"])(this,e),Object(a["a"])(this,"cardGeometry",void 0),Object(a["a"])(this,"cardMaterial",void 0),Object(a["a"])(this,"cardMesh",void 0),Object(a["a"])(this,"stencilId",void 0),Object(a["a"])(this,"cardOutsideScene",void 0),Object(a["a"])(this,"cardInsideScene",void 0),Object(a["a"])(this,"isFocused",void 0),Object(a["a"])(this,"height",void 0),Object(a["a"])(this,"width",void 0),Object(a["a"])(this,"frontWallGeom",void 0),Object(a["a"])(this,"frontWallMesh",void 0),this.stencilId=n,this.height=t,this.width=r,v==i.RECT?this.cardGeometry=new c["G"](r,t):this.cardGeometry=new c["d"](r,64),this.cardMaterial=new c["z"]({color:15790320,stencilWrite:!0,stencilRef:n,stencilZPass:c["M"],depthWrite:!1}),this.cardMesh=new c["y"](this.cardGeometry,this.cardMaterial),this.cardMesh["tardis"]=this,this.cardMesh.position.copy(h),this.cardMesh.rotation.copy(d),this.cardOutsideScene=s,this.cardInsideScene=l,this.cardOutsideScene.add(this.cardMesh),this.isFocused=!1,this.frontWallGeom=new c["c"],this.frontWallMesh=new c["y"]}return Object(s["a"])(e,[{key:"addVisibleObjectGroup",value:function(e){var t=this;this._recursiveGetGroupProperty(e,(function(e){e.stencilWrite=!0,e.stencilRef=t.stencilId,e.stencilFunc=c["m"]})),this.cardOutsideScene.add(e)}},{key:"_recursiveGetGroupProperty",value:function(e,t){var r=this;e.children.forEach((function(e){e instanceof c["y"]?e.material instanceof c["w"]?t(e.material):e.material instanceof Array&&e.material.forEach((function(e){t(e)})):e instanceof c["s"]&&r._recursiveGetGroupProperty(e,t)}))}},{key:"addFrontWall",value:function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:10,t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:void 0,r=[new c["X"](-this.width/2,-this.height/2,0),new c["X"](-this.width/2,this.height/2,0),new c["X"](this.width/2,this.height/2,0),new c["X"](this.width/2,-this.height/2,0),new c["X"](-this.width/2,-this.height/2,-e),new c["X"](-this.width/2,this.height/2,-e),new c["X"](this.width/2,this.height/2,-e),new c["X"](this.width/2,-this.height/2,-e)],i=[1,0,4,5,1,4,2,1,5,6,2,5,3,2,6,7,3,6,0,3,7,4,0,7],n=[];i.forEach((function(e){var t=new c["X"];t.copy(r[e]),n.push(t)})),this.frontWallGeom.setFromPoints(n),this.frontWallGeom.computeVertexNormals();var o=void 0==t?new c["A"]({color:10526880}):t;o.stencilWrite=!0,o.stencilRef=this.stencilId,o.stencilFunc=c["m"],this.frontWallMesh=new c["y"](this.frontWallGeom,o),this.frontWallMesh.position.copy(this.cardMesh.position),this.cardOutsideScene.add(this.frontWallMesh)}},{key:"addSkybox",value:function(e){var t=Object(l["a"])("CardSkyboxShader"),r=Object(n["a"])(t,2),i=r[0],o=r[1],s={tCube:{type:"t",value:e}},a=new c["O"]({uniforms:s,vertexShader:i,fragmentShader:o,side:c["l"],stencilWrite:!0,stencilRef:this.stencilId,stencilZPass:c["M"]});this.cardMaterial=a,this.cardMesh.material=this.cardMaterial}}]),e}()},"26e3":function(e,t,r){"use strict";r.d(t,"a",(function(){return a})),r.d(t,"b",(function(){return c}));var i=r("d4ec"),n=r("bee2"),o=r("ade3"),s=(r("4c53"),r("d3b7"),r("159b"),r("5a89")),a=function(){function e(t){var r=arguments.length>1&&void 0!==arguments[1]?arguments[1]:.5,n=arguments.length>2&&void 0!==arguments[2]?arguments[2]:.5;Object(i["a"])(this,e),Object(o["a"])(this,"damp",void 0),Object(o["a"])(this,"attract",void 0),Object(o["a"])(this,"object",void 0),Object(o["a"])(this,"velocity",void 0),Object(o["a"])(this,"childObjects",void 0),this.damp=r,this.attract=n,this.object=t,this.velocity=new s["X"],this.childObjects=[]}return Object(n["a"])(e,[{key:"addChildObj",value:function(e){this.childObjects.push(e)}},{key:"update",value:function(e){var t=this;this.velocity.multiplyScalar(this.damp);var r=e.sub(this.object.position);this.velocity.add(r.multiplyScalar(this.attract)),this.object.position.add(this.velocity),this.childObjects.forEach((function(e){e.position.copy(t.object.position),e.rotation.copy(t.object.rotation)}))}}]),e}(),c=function(){function e(t){var r=arguments.length>1&&void 0!==arguments[1]?arguments[1]:.5,n=arguments.length>2&&void 0!==arguments[2]?arguments[2]:.5;Object(i["a"])(this,e),Object(o["a"])(this,"damp",void 0),Object(o["a"])(this,"attract",void 0),Object(o["a"])(this,"velocity",void 0),Object(o["a"])(this,"currentVec3",void 0),this.damp=r,this.attract=n,this.velocity=new s["X"],this.currentVec3=t}return Object(n["a"])(e,[{key:"update",value:function(e){this.velocity.multiplyScalar(this.damp);var t=e.sub(this.currentVec3);this.velocity.add(t.multiplyScalar(this.attract)),this.currentVec3.add(this.velocity)}}]),e}()},"32d9":function(e,t,r){"use strict";r.d(t,"a",(function(){return l}));var i=r("5a89"),n={uniforms:{tDiffuse:{value:null},opacity:{value:1}},vertexShader:"\n\n\t\tvarying vec2 vUv;\n\n\t\tvoid main() {\n\n\t\t\tvUv = uv;\n\t\t\tgl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n\n\t\t}",fragmentShader:"\n\n\t\tuniform float opacity;\n\n\t\tuniform sampler2D tDiffuse;\n\n\t\tvarying vec2 vUv;\n\n\t\tvoid main() {\n\n\t\t\tvec4 texel = texture2D( tDiffuse, vUv );\n\t\t\tgl_FragColor = opacity * texel;\n\n\t\t}"},o=r("360d"),s=r("1b53");class a extends s["b"]{constructor(e,t){super(),this.scene=e,this.camera=t,this.clear=!0,this.needsSwap=!1,this.inverse=!1}render(e,t,r){const i=e.getContext(),n=e.state;let o,s;n.buffers.color.setMask(!1),n.buffers.depth.setMask(!1),n.buffers.color.setLocked(!0),n.buffers.depth.setLocked(!0),this.inverse?(o=0,s=1):(o=1,s=0),n.buffers.stencil.setTest(!0),n.buffers.stencil.setOp(i.REPLACE,i.REPLACE,i.REPLACE),n.buffers.stencil.setFunc(i.ALWAYS,o,4294967295),n.buffers.stencil.setClear(s),n.buffers.stencil.setLocked(!0),e.setRenderTarget(r),this.clear&&e.clear(),e.render(this.scene,this.camera),e.setRenderTarget(t),this.clear&&e.clear(),e.render(this.scene,this.camera),n.buffers.color.setLocked(!1),n.buffers.depth.setLocked(!1),n.buffers.stencil.setLocked(!1),n.buffers.stencil.setFunc(i.EQUAL,1,4294967295),n.buffers.stencil.setOp(i.KEEP,i.KEEP,i.KEEP),n.buffers.stencil.setLocked(!0)}}class c extends s["b"]{constructor(){super(),this.needsSwap=!1}render(e){e.state.buffers.stencil.setLocked(!1),e.state.buffers.stencil.setTest(!1)}}class l{constructor(e,t){if(this.renderer=e,void 0===t){const r={minFilter:i["u"],magFilter:i["u"],format:i["J"]},n=e.getSize(new i["W"]);this._pixelRatio=e.getPixelRatio(),this._width=n.width,this._height=n.height,t=new i["Z"](this._width*this._pixelRatio,this._height*this._pixelRatio,r),t.texture.name="EffectComposer.rt1"}else this._pixelRatio=1,this._width=t.width,this._height=t.height;this.renderTarget1=t,this.renderTarget2=t.clone(),this.renderTarget2.texture.name="EffectComposer.rt2",this.writeBuffer=this.renderTarget1,this.readBuffer=this.renderTarget2,this.renderToScreen=!0,this.passes=[],void 0===n&&console.error("THREE.EffectComposer relies on CopyShader"),void 0===o["a"]&&console.error("THREE.EffectComposer relies on ShaderPass"),this.copyPass=new o["a"](n),this.clock=new i["e"]}swapBuffers(){const e=this.readBuffer;this.readBuffer=this.writeBuffer,this.writeBuffer=e}addPass(e){this.passes.push(e),e.setSize(this._width*this._pixelRatio,this._height*this._pixelRatio)}insertPass(e,t){this.passes.splice(t,0,e),e.setSize(this._width*this._pixelRatio,this._height*this._pixelRatio)}removePass(e){const t=this.passes.indexOf(e);-1!==t&&this.passes.splice(t,1)}isLastEnabledPass(e){for(let t=e+1;t<this.passes.length;t++)if(this.passes[t].enabled)return!1;return!0}render(e){void 0===e&&(e=this.clock.getDelta());const t=this.renderer.getRenderTarget();let r=!1;for(let i=0,n=this.passes.length;i<n;i++){const t=this.passes[i];if(!1!==t.enabled){if(t.renderToScreen=this.renderToScreen&&this.isLastEnabledPass(i),t.render(this.renderer,this.writeBuffer,this.readBuffer,e,r),t.needsSwap){if(r){const t=this.renderer.getContext(),r=this.renderer.state.buffers.stencil;r.setFunc(t.NOTEQUAL,1,4294967295),this.copyPass.render(this.renderer,this.writeBuffer,this.readBuffer,e),r.setFunc(t.EQUAL,1,4294967295)}this.swapBuffers()}void 0!==a&&(t instanceof a?r=!0:t instanceof c&&(r=!1))}}this.renderer.setRenderTarget(t)}reset(e){if(void 0===e){const t=this.renderer.getSize(new i["W"]);this._pixelRatio=this.renderer.getPixelRatio(),this._width=t.width,this._height=t.height,e=this.renderTarget1.clone(),e.setSize(this._width*this._pixelRatio,this._height*this._pixelRatio)}this.renderTarget1.dispose(),this.renderTarget2.dispose(),this.renderTarget1=e,this.renderTarget2=e.clone(),this.writeBuffer=this.renderTarget1,this.readBuffer=this.renderTarget2}setSize(e,t){this._width=e,this._height=t;const r=this._width*this._pixelRatio,i=this._height*this._pixelRatio;this.renderTarget1.setSize(r,i),this.renderTarget2.setSize(r,i);for(let n=0;n<this.passes.length;n++)this.passes[n].setSize(r,i)}setPixelRatio(e){this._pixelRatio=e,this.setSize(this._width,this._height)}}new i["D"](-1,1,1,-1,0,1);const h=new i["c"];h.setAttribute("position",new i["p"]([-1,3,0,-1,-1,0,3,-1,0],3)),h.setAttribute("uv",new i["p"]([0,2,0,0,2,0],2))},"360d":function(e,t,r){"use strict";r.d(t,"a",(function(){return o}));var i=r("5a89"),n=r("1b53");class o extends n["b"]{constructor(e,t){super(),this.textureID=void 0!==t?t:"tDiffuse",e instanceof i["O"]?(this.uniforms=e.uniforms,this.material=e):e&&(this.uniforms=i["U"].clone(e.uniforms),this.material=new i["O"]({defines:Object.assign({},e.defines),uniforms:this.uniforms,vertexShader:e.vertexShader,fragmentShader:e.fragmentShader})),this.fsQuad=new n["a"](this.material)}render(e,t,r){this.uniforms[this.textureID]&&(this.uniforms[this.textureID].value=r.texture),this.fsQuad.material=this.material,this.renderToScreen?(e.setRenderTarget(null),this.fsQuad.render(e)):(e.setRenderTarget(t),this.clear&&e.clear(e.autoClearColor,e.autoClearDepth,e.autoClearStencil),this.fsQuad.render(e))}}},"4c53":function(e,t,r){"use strict";var i=r("23e7"),n=r("857a"),o=r("af03");i({target:"String",proto:!0,forced:o("sub")},{sub:function(){return n(this,"sub","","")}})},"4e0a":function(e,t,r){"use strict";function i(e){try{var t=r("5d16")("./"+e+"/V.glsl"),i=r("8e1c")("./"+e+"/F.glsl");return[t,i]}catch(n){return console.error("failed to fetch shader",n),["",""]}}r.d(t,"a",(function(){return i}))},"5d16":function(e,t,r){var i={"./CardSkyboxShader/V.glsl":"8437","./FresnelShader/V.glsl":"8c9f","./GlassFrontShader/V.glsl":"c182"};function n(e){var t=o(e);return r(t)}function o(e){if(!r.o(i,e)){var t=new Error("Cannot find module '"+e+"'");throw t.code="MODULE_NOT_FOUND",t}return i[e]}n.keys=function(){return Object.keys(i)},n.resolve=o,e.exports=n,n.id="5d16"},"71cc":function(e,t){e.exports="uniform vec2 resolution;\r\nuniform sampler2D tBackground;\r\nuniform sampler2D tBackDepth;\r\n\r\nvarying vec2 vUv;\r\nvarying vec3 vNormal;\r\nvarying vec3 vCameraRay;\r\nvarying vec4 vWorldPosition;\r\nvarying mat4 vProjectionMatrix;\r\n\r\nfloat near = 0.1;\r\nfloat far  = 20000.0; \r\n\r\nfloat LinearizeDepth(float depth) \r\n{\r\n    float z = depth * 2.0 - 1.0; // back to NDC \r\n    return (2.0 * near * far) / (far + near - z * (far - near));\t\r\n}\r\n\r\nfloat refractionFactor = 1.49; // glass\r\n\r\nvoid main() {\r\n\r\n\r\n    float backDepth = float(texture2D( tBackDepth, vec2( gl_FragCoord.x / resolution.x, gl_FragCoord.y / resolution.y ) ).x);\r\n    float frontDepth = gl_FragCoord.z;\r\n\r\n    float backDepthlinear = LinearizeDepth(backDepth);\r\n    float frontDepthlinear = LinearizeDepth(frontDepth);\r\n    \r\n    float backToFrontDepth = backDepthlinear - frontDepthlinear;\r\n    backToFrontDepth *= 0.005;\r\n    \r\n    vec3 vRefract = refract( vCameraRay, vNormal, 1.0 / refractionFactor );\r\n    vec4 vOriginalPos = vWorldPosition + vec4(vCameraRay, 0.0);\r\n    vec4 vRefractedPos = vWorldPosition + vec4(vRefract, 0.0);\r\n    // vRefractedPos = vOriginalPos; //vWorldPosition + vec4(vRefract, 0.0);\r\n    vec4 vOriginalProjectedPos = vProjectionMatrix * vOriginalPos;\r\n    vec4 vRefractedProjectedPos = vProjectionMatrix * vRefractedPos;\r\n    vec4 offset = normalize(vRefractedProjectedPos - vOriginalProjectedPos);\r\n\r\n    // snell's law\r\n    float cosTheta1 = dot(vCameraRay, vNormal);\r\n    float sinTheta1 = 1.0 - cosTheta1 * cosTheta1;\r\n    float sinTheta2 = sinTheta1 / refractionFactor;\r\n    float theta2 = asin(sinTheta2);\r\n    offset *= theta2;\r\n\r\n    float samplePointX = gl_FragCoord.x / resolution.x + backToFrontDepth * offset.x;\r\n    float samplePointY = gl_FragCoord.y / resolution.y + backToFrontDepth * offset.y;\r\n    \r\n    vec3 backColor = texture2D( tBackground, vec2( samplePointX, samplePointY ) ).xyz;\r\n    // vec4 vNewSample = projectionMatrix * (vec4(vRefract, 0.0) + worldPosition);\r\n    // vRefractOffset = vNewSample.xy / vNewSample.w;\r\n    // // vNormal = normal;\r\n    // vec3 vRefractOffsetT = vRefract + 0.5;\r\n    // gl_FragColor = vec4(backColor * 0.5 + vNormal * 0.2 + depthColor * 0.1, 1.0);\r\n    gl_FragColor = vec4(backColor, 1.0);\r\n    // gl_FragColor = vec4(vec3(backToFrontDepth) * 100.0, 1.0);\r\n    // gl_FragColor = vec4(offset.xy / offset.w + 0.5, 0.0, 1.0);\r\n    // gl_FragColor = vec4(vec3(theta2), 1.0);\r\n\r\n    // gl_FragColor = vec4(vec3(gl_FragCoord.z / gl_FragCoord.w - fragColor2.x * 10.0) / 2000.0, 1.0);\r\n}"},"7bcb":function(e,t){e.exports="uniform samplerCube tCube;\r\nvarying vec3 vReflect;\r\nvarying vec3 vRefract[3];\r\nvarying float vReflectionFactor;\r\nvarying vec2 vUv;\r\n\r\nuniform sampler2D tNormal;\r\n\r\nvoid main() {\r\n\r\n    vec4 reflectedColor = textureCube( tCube, vec3( -vReflect.x, vReflect.yz ) );\r\n    vec4 refractedColor = vec4( 1.0 );\r\n\r\n    vec3 tnm = vec3(texture2D(tNormal, vUv));\r\n    tnm = tnm * 2.0 - 1.0;\r\n    tnm = 0.0 * tnm;\r\n    // tnm *= 0.01;\r\n    refractedColor.r = textureCube( tCube, vec3( -vRefract[0].x, vRefract[0].yz ) - tnm ).r;\r\n    refractedColor.g = textureCube( tCube, vec3( -vRefract[1].x, vRefract[1].yz ) - tnm ).g;\r\n    refractedColor.b = textureCube( tCube, vec3( -vRefract[2].x, vRefract[2].yz ) - tnm ).b;\r\n\r\n    gl_FragColor = mix( refractedColor, reflectedColor, clamp( vReflectionFactor, 0.0, 1.0 ) );\r\n}"},8437:function(e,t){e.exports="varying vec3 vPos;\r\n\r\nvoid main() {\r\n    vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );\r\n    gl_Position = projectionMatrix * mvPosition;\r\n    vec4 worldPosition = modelMatrix * vec4( position, 1.0 );\r\n    vPos = worldPosition.xyz - cameraPosition;\r\n}\r\n"},"857a":function(e,t,r){var i=r("e330"),n=r("1d80"),o=r("577e"),s=/"/g,a=i("".replace);e.exports=function(e,t,r,i){var c=o(n(e)),l="<"+t;return""!==r&&(l+=" "+r+'="'+a(o(i),s,"&quot;")+'"'),l+">"+c+"</"+t+">"}},"8c9f":function(e,t){e.exports="uniform float mRefractionRatio;\r\nuniform float mFresnelBias;\r\nuniform float mFresnelScale;\r\nuniform float mFresnelPower;\r\n\r\nvarying vec3 vReflect;\r\nvarying vec3 vRefract[3];\r\nvarying float vReflectionFactor;\r\nvarying vec2 vUv;\r\n\r\nvoid main() {\r\n\r\n    vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );\r\n    vec4 worldPosition = modelMatrix * vec4( position, 1.0 );\r\n\r\n    vec3 worldNormal = normalize( mat3( modelMatrix[0].xyz, modelMatrix[1].xyz, modelMatrix[2].xyz ) * normal );\r\n\r\n    vec3 I = worldPosition.xyz - cameraPosition;\r\n\r\n    vReflect = reflect( I, worldNormal );\r\n    vRefract[0] = refract( normalize( I ), worldNormal, mRefractionRatio );\r\n    vRefract[1] = refract( normalize( I ), worldNormal, mRefractionRatio * 0.99 );\r\n    vRefract[2] = refract( normalize( I ), worldNormal, mRefractionRatio * 0.98 );\r\n    vReflectionFactor = mFresnelBias + mFresnelScale * pow( 1.0 + dot( normalize( I ), worldNormal ), mFresnelPower );\r\n\r\n    gl_Position = projectionMatrix * mvPosition;\r\n    vUv = uv;\r\n\r\n}\r\n"},"8e1c":function(e,t,r){var i={"./CardSkyboxShader/F.glsl":"91bc","./FresnelShader/F.glsl":"7bcb","./GlassFrontShader/F.glsl":"71cc"};function n(e){var t=o(e);return r(t)}function o(e){if(!r.o(i,e)){var t=new Error("Cannot find module '"+e+"'");throw t.code="MODULE_NOT_FOUND",t}return i[e]}n.keys=function(){return Object.keys(i)},n.resolve=o,e.exports=n,n.id="8e1c"},"91bc":function(e,t){e.exports="uniform samplerCube tCube;\r\nvarying vec3 vPos;\r\n\r\nvoid main() {\r\n    vec4 reflectedColor = textureCube( tCube, vec3(-vPos.x, vPos.yz) );\r\n    gl_FragColor = reflectedColor;\r\n}"},"93e9":function(e,t,r){"use strict";r.d(t,"a",(function(){return o}));var i=r("5a89"),n=r("1b53");class o extends n["b"]{constructor(e,t,r,n,o){super(),this.scene=e,this.camera=t,this.overrideMaterial=r,this.clearColor=n,this.clearAlpha=void 0!==o?o:0,this.clear=!0,this.clearDepth=!1,this.needsSwap=!1,this._oldClearColor=new i["f"]}render(e,t,r){const i=e.autoClear;let n,o;e.autoClear=!1,void 0!==this.overrideMaterial&&(o=this.scene.overrideMaterial,this.scene.overrideMaterial=this.overrideMaterial),this.clearColor&&(e.getClearColor(this._oldClearColor),n=e.getClearAlpha(),e.setClearColor(this.clearColor,this.clearAlpha)),this.clearDepth&&e.clearDepth(),e.setRenderTarget(this.renderToScreen?null:r),this.clear&&e.clear(e.autoClearColor,e.autoClearDepth,e.autoClearStencil),e.render(this.scene,this.camera),this.clearColor&&e.setClearColor(this._oldClearColor,n),void 0!==this.overrideMaterial&&(this.scene.overrideMaterial=o),e.autoClear=i}}},af03:function(e,t,r){var i=r("d039");e.exports=function(e){return i((function(){var t=""[e]('"');return t!==t.toLowerCase()||t.split('"').length>3}))}},c182:function(e,t){e.exports="varying vec2 vUv;\r\nvarying vec3 vNormal;\r\nvarying vec3 vCameraRay;\r\nvarying vec4 vWorldPosition;\r\nvarying mat4 vProjectionMatrix;\r\n// varying vec3 vRefract;\r\n// varying vec2 vRefractOffset;\r\n\r\n\r\nvoid main() {\r\n    vWorldPosition = modelMatrix * vec4( position, 1.0 );\r\n    vec4 mvPosition = viewMatrix * vWorldPosition;\r\n    gl_Position = projectionMatrix * mvPosition;\r\n    vUv = uv;\r\n\r\n    vNormal = normalize( mat3( modelMatrix[0].xyz, modelMatrix[1].xyz, modelMatrix[2].xyz ) * normal );\r\n    vCameraRay = normalize(vWorldPosition.xyz - cameraPosition);\r\n\r\n    vProjectionMatrix = projectionMatrix * modelViewMatrix;\r\n}\r\n"}}]);
//# sourceMappingURL=chunk-1744ab30.2e969810.js.map