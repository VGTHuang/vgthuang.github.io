(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-2e2f83ce"],{"00bc":function(e,t,n){"use strict";n("c296")},4721:function(e,t,n){"use strict";n.d(t,"a",(function(){return r}));var o=n("5a89");const a={type:"change"},i={type:"start"},s={type:"end"};class r extends o["q"]{constructor(e,t){super(),void 0===t&&console.warn('THREE.OrbitControls: The second parameter "domElement" is now mandatory.'),t===document&&console.error('THREE.OrbitControls: "document" should not be used as the target "domElement". Please use "renderer.domElement" instead.'),this.object=e,this.domElement=t,this.domElement.style.touchAction="none",this.enabled=!0,this.target=new o["ib"],this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"ArrowLeft",UP:"ArrowUp",RIGHT:"ArrowRight",BOTTOM:"ArrowDown"},this.mouseButtons={LEFT:o["C"].ROTATE,MIDDLE:o["C"].DOLLY,RIGHT:o["C"].PAN},this.touches={ONE:o["bb"].ROTATE,TWO:o["bb"].DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._domElementKeyEvents=null,this.getPolarAngle=function(){return d.phi},this.getAzimuthalAngle=function(){return d.theta},this.getDistance=function(){return this.object.position.distanceTo(this.target)},this.listenToKeyEvents=function(e){e.addEventListener("keydown",se),this._domElementKeyEvents=e},this.saveState=function(){n.target0.copy(n.target),n.position0.copy(n.object.position),n.zoom0=n.object.zoom},this.reset=function(){n.target.copy(n.target0),n.object.position.copy(n.position0),n.object.zoom=n.zoom0,n.object.updateProjectionMatrix(),n.dispatchEvent(a),n.update(),c=r.NONE},this.update=function(){const t=new o["ib"],i=(new o["R"]).setFromUnitVectors(e.up,new o["ib"](0,1,0)),s=i.clone().invert(),b=new o["ib"],g=new o["R"],f=2*Math.PI;return function(){const e=n.object.position;t.copy(e).sub(n.target),t.applyQuaternion(i),d.setFromVector3(t),n.autoRotate&&c===r.NONE&&k(L()),n.enableDamping?(d.theta+=h.theta*n.dampingFactor,d.phi+=h.phi*n.dampingFactor):(d.theta+=h.theta,d.phi+=h.phi);let o=n.minAzimuthAngle,E=n.maxAzimuthAngle;return isFinite(o)&&isFinite(E)&&(o<-Math.PI?o+=f:o>Math.PI&&(o-=f),E<-Math.PI?E+=f:E>Math.PI&&(E-=f),d.theta=o<=E?Math.max(o,Math.min(E,d.theta)):d.theta>(o+E)/2?Math.max(o,d.theta):Math.min(E,d.theta)),d.phi=Math.max(n.minPolarAngle,Math.min(n.maxPolarAngle,d.phi)),d.makeSafe(),d.radius*=u,d.radius=Math.max(n.minDistance,Math.min(n.maxDistance,d.radius)),!0===n.enableDamping?n.target.addScaledVector(m,n.dampingFactor):n.target.add(m),t.setFromSpherical(d),t.applyQuaternion(s),e.copy(n.target).add(t),n.object.lookAt(n.target),!0===n.enableDamping?(h.theta*=1-n.dampingFactor,h.phi*=1-n.dampingFactor,m.multiplyScalar(1-n.dampingFactor)):(h.set(0,0,0),m.set(0,0,0)),u=1,!!(p||b.distanceToSquared(n.object.position)>l||8*(1-g.dot(n.object.quaternion))>l)&&(n.dispatchEvent(a),b.copy(n.object.position),g.copy(n.object.quaternion),p=!1,!0)}}(),this.dispose=function(){n.domElement.removeEventListener("contextmenu",le),n.domElement.removeEventListener("pointerdown",Q),n.domElement.removeEventListener("pointercancel",ne),n.domElement.removeEventListener("wheel",ie),n.domElement.removeEventListener("pointermove",ee),n.domElement.removeEventListener("pointerup",te),null!==n._domElementKeyEvents&&n._domElementKeyEvents.removeEventListener("keydown",se)};const n=this,r={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6};let c=r.NONE;const l=1e-6,d=new o["ab"],h=new o["ab"];let u=1;const m=new o["ib"];let p=!1;const b=new o["hb"],g=new o["hb"],f=new o["hb"],E=new o["hb"],w=new o["hb"],v=new o["hb"],O=new o["hb"],y=new o["hb"],P=new o["hb"],T=[],j={};function L(){return 2*Math.PI/60/60*n.autoRotateSpeed}function A(){return Math.pow(.95,n.zoomSpeed)}function k(e){h.theta-=e}function N(e){h.phi-=e}const x=function(){const e=new o["ib"];return function(t,n){e.setFromMatrixColumn(n,0),e.multiplyScalar(-t),m.add(e)}}(),M=function(){const e=new o["ib"];return function(t,o){!0===n.screenSpacePanning?e.setFromMatrixColumn(o,1):(e.setFromMatrixColumn(o,0),e.crossVectors(n.object.up,e)),e.multiplyScalar(t),m.add(e)}}(),S=function(){const e=new o["ib"];return function(t,o){const a=n.domElement;if(n.object.isPerspectiveCamera){const i=n.object.position;e.copy(i).sub(n.target);let s=e.length();s*=Math.tan(n.object.fov/2*Math.PI/180),x(2*t*s/a.clientHeight,n.object.matrix),M(2*o*s/a.clientHeight,n.object.matrix)}else n.object.isOrthographicCamera?(x(t*(n.object.right-n.object.left)/n.object.zoom/a.clientWidth,n.object.matrix),M(o*(n.object.top-n.object.bottom)/n.object.zoom/a.clientHeight,n.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),n.enablePan=!1)}}();function R(e){n.object.isPerspectiveCamera?u/=e:n.object.isOrthographicCamera?(n.object.zoom=Math.max(n.minZoom,Math.min(n.maxZoom,n.object.zoom*e)),n.object.updateProjectionMatrix(),p=!0):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),n.enableZoom=!1)}function C(e){n.object.isPerspectiveCamera?u*=e:n.object.isOrthographicCamera?(n.object.zoom=Math.max(n.minZoom,Math.min(n.maxZoom,n.object.zoom/e)),n.object.updateProjectionMatrix(),p=!0):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),n.enableZoom=!1)}function Y(e){b.set(e.clientX,e.clientY)}function I(e){O.set(e.clientX,e.clientY)}function D(e){E.set(e.clientX,e.clientY)}function z(e){g.set(e.clientX,e.clientY),f.subVectors(g,b).multiplyScalar(n.rotateSpeed);const t=n.domElement;k(2*Math.PI*f.x/t.clientHeight),N(2*Math.PI*f.y/t.clientHeight),b.copy(g),n.update()}function H(e){y.set(e.clientX,e.clientY),P.subVectors(y,O),P.y>0?R(A()):P.y<0&&C(A()),O.copy(y),n.update()}function _(e){w.set(e.clientX,e.clientY),v.subVectors(w,E).multiplyScalar(n.panSpeed),S(v.x,v.y),E.copy(w),n.update()}function X(e){e.deltaY<0?C(A()):e.deltaY>0&&R(A()),n.update()}function F(e){let t=!1;switch(e.code){case n.keys.UP:S(0,n.keyPanSpeed),t=!0;break;case n.keys.BOTTOM:S(0,-n.keyPanSpeed),t=!0;break;case n.keys.LEFT:S(n.keyPanSpeed,0),t=!0;break;case n.keys.RIGHT:S(-n.keyPanSpeed,0),t=!0;break}t&&(e.preventDefault(),n.update())}function Z(){if(1===T.length)b.set(T[0].pageX,T[0].pageY);else{const e=.5*(T[0].pageX+T[1].pageX),t=.5*(T[0].pageY+T[1].pageY);b.set(e,t)}}function U(){if(1===T.length)E.set(T[0].pageX,T[0].pageY);else{const e=.5*(T[0].pageX+T[1].pageX),t=.5*(T[0].pageY+T[1].pageY);E.set(e,t)}}function W(){const e=T[0].pageX-T[1].pageX,t=T[0].pageY-T[1].pageY,n=Math.sqrt(e*e+t*t);O.set(0,n)}function K(){n.enableZoom&&W(),n.enablePan&&U()}function V(){n.enableZoom&&W(),n.enableRotate&&Z()}function q(e){if(1==T.length)g.set(e.pageX,e.pageY);else{const t=me(e),n=.5*(e.pageX+t.x),o=.5*(e.pageY+t.y);g.set(n,o)}f.subVectors(g,b).multiplyScalar(n.rotateSpeed);const t=n.domElement;k(2*Math.PI*f.x/t.clientHeight),N(2*Math.PI*f.y/t.clientHeight),b.copy(g)}function B(e){if(1===T.length)w.set(e.pageX,e.pageY);else{const t=me(e),n=.5*(e.pageX+t.x),o=.5*(e.pageY+t.y);w.set(n,o)}v.subVectors(w,E).multiplyScalar(n.panSpeed),S(v.x,v.y),E.copy(w)}function G(e){const t=me(e),o=e.pageX-t.x,a=e.pageY-t.y,i=Math.sqrt(o*o+a*a);y.set(0,i),P.set(0,Math.pow(y.y/O.y,n.zoomSpeed)),R(P.y),O.copy(y)}function $(e){n.enableZoom&&G(e),n.enablePan&&B(e)}function J(e){n.enableZoom&&G(e),n.enableRotate&&q(e)}function Q(e){!1!==n.enabled&&(0===T.length&&(n.domElement.setPointerCapture(e.pointerId),n.domElement.addEventListener("pointermove",ee),n.domElement.addEventListener("pointerup",te)),de(e),"touch"===e.pointerType?re(e):oe(e))}function ee(e){!1!==n.enabled&&("touch"===e.pointerType?ce(e):ae(e))}function te(e){he(e),0===T.length&&(n.domElement.releasePointerCapture(e.pointerId),n.domElement.removeEventListener("pointermove",ee),n.domElement.removeEventListener("pointerup",te)),n.dispatchEvent(s),c=r.NONE}function ne(e){he(e)}function oe(e){let t;switch(e.button){case 0:t=n.mouseButtons.LEFT;break;case 1:t=n.mouseButtons.MIDDLE;break;case 2:t=n.mouseButtons.RIGHT;break;default:t=-1}switch(t){case o["C"].DOLLY:if(!1===n.enableZoom)return;I(e),c=r.DOLLY;break;case o["C"].ROTATE:if(e.ctrlKey||e.metaKey||e.shiftKey){if(!1===n.enablePan)return;D(e),c=r.PAN}else{if(!1===n.enableRotate)return;Y(e),c=r.ROTATE}break;case o["C"].PAN:if(e.ctrlKey||e.metaKey||e.shiftKey){if(!1===n.enableRotate)return;Y(e),c=r.ROTATE}else{if(!1===n.enablePan)return;D(e),c=r.PAN}break;default:c=r.NONE}c!==r.NONE&&n.dispatchEvent(i)}function ae(e){if(!1!==n.enabled)switch(c){case r.ROTATE:if(!1===n.enableRotate)return;z(e);break;case r.DOLLY:if(!1===n.enableZoom)return;H(e);break;case r.PAN:if(!1===n.enablePan)return;_(e);break}}function ie(e){!1!==n.enabled&&!1!==n.enableZoom&&c===r.NONE&&(e.preventDefault(),n.dispatchEvent(i),X(e),n.dispatchEvent(s))}function se(e){!1!==n.enabled&&!1!==n.enablePan&&F(e)}function re(e){switch(ue(e),T.length){case 1:switch(n.touches.ONE){case o["bb"].ROTATE:if(!1===n.enableRotate)return;Z(),c=r.TOUCH_ROTATE;break;case o["bb"].PAN:if(!1===n.enablePan)return;U(),c=r.TOUCH_PAN;break;default:c=r.NONE}break;case 2:switch(n.touches.TWO){case o["bb"].DOLLY_PAN:if(!1===n.enableZoom&&!1===n.enablePan)return;K(),c=r.TOUCH_DOLLY_PAN;break;case o["bb"].DOLLY_ROTATE:if(!1===n.enableZoom&&!1===n.enableRotate)return;V(),c=r.TOUCH_DOLLY_ROTATE;break;default:c=r.NONE}break;default:c=r.NONE}c!==r.NONE&&n.dispatchEvent(i)}function ce(e){switch(ue(e),c){case r.TOUCH_ROTATE:if(!1===n.enableRotate)return;q(e),n.update();break;case r.TOUCH_PAN:if(!1===n.enablePan)return;B(e),n.update();break;case r.TOUCH_DOLLY_PAN:if(!1===n.enableZoom&&!1===n.enablePan)return;$(e),n.update();break;case r.TOUCH_DOLLY_ROTATE:if(!1===n.enableZoom&&!1===n.enableRotate)return;J(e),n.update();break;default:c=r.NONE}}function le(e){!1!==n.enabled&&e.preventDefault()}function de(e){T.push(e)}function he(e){delete j[e.pointerId];for(let t=0;t<T.length;t++)if(T[t].pointerId==e.pointerId)return void T.splice(t,1)}function ue(e){let t=j[e.pointerId];void 0===t&&(t=new o["hb"],j[e.pointerId]=t),t.set(e.pageX,e.pageY)}function me(e){const t=e.pointerId===T[0].pointerId?T[1]:T[0];return j[t.pointerId]}n.domElement.addEventListener("contextmenu",le),n.domElement.addEventListener("pointerdown",Q),n.domElement.addEventListener("pointercancel",ne),n.domElement.addEventListener("wheel",ie,{passive:!1}),this.update()}}},"5b19":function(e,t,n){"use strict";n.r(t);var o=function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("div",[n("div",{ref:"canvas-container",attrs:{id:"canvas-container"}},[n("div",{staticClass:"loading-container",class:{"loading-container--hidden":e.loadingStat.isLoaded,"loading-container--none":e.loadingStat.noLoadingStat}},[n("div",{staticClass:"load-stat"},[n("div",{staticClass:"load-pgbar",style:{width:100*e.loadingStat.loaded/e.loadingStat.total+"%"}}),n("div",{staticClass:"load-text"},[e._v(e._s(Math.round(100*e.loadingStat.loaded/e.loadingStat.total))+" %")])])])])])},a=[],i=n("2b0e"),s=n("d4ec"),r=n("bee2"),c=n("ade3"),l=n("5a89"),d=n("61d9"),h=n("4721"),u=function(){function e(t,n,o,a){var i=this;Object(s["a"])(this,e),Object(c["a"])(this,"container",void 0),Object(c["a"])(this,"renderer",void 0),Object(c["a"])(this,"scene",void 0),Object(c["a"])(this,"stats",void 0),Object(c["a"])(this,"controls",void 0),Object(c["a"])(this,"mousePosX",void 0),Object(c["a"])(this,"mousePosY",void 0),Object(c["a"])(this,"pointer",void 0),Object(c["a"])(this,"camera",void 0),Object(c["a"])(this,"loadingManager",void 0),Object(c["a"])(this,"animate",(function(){requestAnimationFrame(i.animate),i.render(),i.stats.update(),i.controls.update()}));var r=this;this.container=t,this.camera=new l["M"](45,window.innerWidth/window.innerHeight,1,5e3),this.renderer=new l["mb"]({alpha:!0}),this.renderer.outputEncoding=l["nb"],this.renderer.toneMappingExposure=l["a"],this.renderer.toneMappingExposure=1.2,this.renderer.setClearColor(16777215),this.renderer.setPixelRatio(window.devicePixelRatio),this.renderer.setSize(window.innerWidth,window.innerHeight),this.renderer.domElement.style.position="absolute",this.renderer.domElement.style.top="0px",this.renderer.domElement.style.left="0px",this.container.appendChild(this.renderer.domElement),this.stats=new d["Stats"],this.stats.domElement.style.zIndex="100",this.container.appendChild(this.stats.domElement),this.controls=new h["a"](this.camera,this.container),this.camera.position.set(0,200,150),this.scene=new l["W"],this.loadingManager=new l["B"]((function(){n&&n()}),(function(e,t,n){o&&o(e,t,n)}),(function(e){a&&a(e)}));var u="./cube2/",m=[u+"px.png",u+"nx.png",u+"py.png",u+"ny.png",u+"pz.png",u+"nz.png"],p=new l["j"](this.loadingManager).load(m);p.mapping=l["i"],this.scene.environment=p;var b=new l["m"](16777215,.5);this.scene.add(b),this.mousePosX=0,this.mousePosY=0,this.pointer=new l["hb"],window.addEventListener("resize",(function(){r.onWindowResize()})),window.addEventListener("click",(function(e){r.onClick(e)})),this.animate()}return Object(r["a"])(e,[{key:"render",value:function(){this.renderer.render(this.scene,this.camera)}},{key:"onWindowResize",value:function(){this.camera.aspect=window.innerWidth/window.innerHeight,this.camera.updateProjectionMatrix(),this.renderer.setSize(window.innerWidth,window.innerHeight)}},{key:"onClick",value:function(e){}}]),e}(),m=i["a"].extend({data:function(){return{fpm:null,loadingStat:{loaded:0,total:1,isLoaded:!1,noLoadingStat:!1}}},mounted:function(){var e=this,t=this;this.$refs["canvas-container"]instanceof HTMLElement&&(this.fpm=new u(this.$refs["canvas-container"],(function(){console.log("loaded"),t.loadingStat.isLoaded=!0,setTimeout((function(){t.loadingStat.noLoadingStat=!0}),501)}),(function(t,n,o){e.loadingStat.loaded=n,e.loadingStat.total=o}),(function(e){console.error("failed to load asset: "+e),t.loadingStat.isLoaded=!0})))}}),p=m,b=(n("00bc"),n("2877")),g=Object(b["a"])(p,o,a,!1,null,"17a177c4",null);t["default"]=g.exports},c296:function(e,t,n){}}]);
//# sourceMappingURL=chunk-2e2f83ce.42c3ab98.js.map